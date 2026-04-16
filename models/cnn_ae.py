# standard library
import os
import sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

# project root setup
notebook_dir = Path.cwd()
project_root = notebook_dir.parent
sys.path.insert(0, str(project_root))

# local imports
from utils.checkpoint import ModelCheckpoint
from utils.losses import (
    masked_l1_loss,
    masked_l1_grad_loss,
    masked_huber_loss,
    masked_multires_l1_loss,
    masked_mae,
    masked_rmse,
    full_mae,
    full_rmse,
    psnr
)

"""
Usage: 
--------------------------------------------------------------------------
- Create Model:
model = CNNAutoencoder(base_channels=12).to(device)
--------------------------------------------------------------------------
- Get Loss function:
lossfn = model.get_loss(
    variant="long_gap",
    shortgap_loss="masked_l1",
    longgap_loss="masked_multires_l1"
)
print(lossfn)
--------------------------------------------------------------------------
- Build optimiser:
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
--------------------------------------------------------------------------
- Compile model:
model.compile(
    optimiser=optimiser,
    lossfn=lossfn,
    device=device,
    use_amp=True,
    use_scheduler=True,
    scheduler_type="plateau",
    scheduler_factor=0.5,
    scheduler_patience=2,
    scheduler_min_lr=1e-7,
)
--------------------------------------------------------------------------
- Train model:
results = model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    n_epochs=50,
    checkpoint_dir=checkpoint_dir,
    history_path=history_dir / "history.csv",
    monitor="val_gap_rmse",
    mode="min",
    patience=6,
    min_delta=1e-4,
    save_best_after_epoch=2,
    grad_clip=1.0,
    verbose=True,
)
"""

# helper for groupnorm
def make_norm(channels, max_groups=8):
    """
    create a GroupNorm layer with a valid number of groups.
    """
    groups = min(max_groups, channels)

    # decrease groups until channels is divisible by groups
    while channels % groups != 0 and groups > 1:
        groups -= 1

    return nn.GroupNorm(groups, channels)


# se block
class SEBlock(nn.Module):
    """
    squeeze-and-excitation block:
    - summarise each channel with global average pooling
    - learn channel importance weights
    - rescale channels adaptively
    """
    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        scale = self.pool(x)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


# MBConv block
class MBConvBlock(nn.Module):
    """
    efficient mobile inverted bottleneck block:
    - optional expansion
    - depthwise convolution
    - SE attention
    - projection back to target width
    - residual connection when shapes match
    """
    def __init__(self, in_channels, out_channels, expansion=4, kernel_size=3, dilation=1, se_reduction=8):
        super().__init__()

        self.use_residual = (in_channels == out_channels)
        self.use_expand = expansion > 1
        hidden_dim = in_channels * expansion if self.use_expand else in_channels

        if self.use_expand:
            self.expand = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
            self.norm1 = make_norm(hidden_dim)

        padding = (kernel_size // 2) * dilation

        self.depthwise = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden_dim,
            bias=False
        )
        self.norm2 = make_norm(hidden_dim)

        self.se = SEBlock(hidden_dim, reduction=se_reduction)

        self.project = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.norm3 = make_norm(out_channels)

    def forward(self, x):
        out = x

        if self.use_expand:
            out = self.expand(x)
            out = F.silu(self.norm1(out))

        out = self.depthwise(out)
        out = F.silu(self.norm2(out))
        out = self.se(out)
        out = self.project(out)
        out = self.norm3(out)

        if self.use_residual:
            out = out + x

        return F.silu(out)


# lightweight residual convolution block
class LightConvBlock(nn.Module):
    """
    cheaper residual block for high-resolution stages
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.use_residual = (in_channels == out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = make_norm(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = make_norm(out_channels)

        if not self.use_residual:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = F.silu(self.norm1(out))
        out = self.conv2(out)
        out = self.norm2(out)

        if self.use_residual:
            out = out + identity
        else:
            out = out + self.skip(identity)

        return F.silu(out)


# temporal context block
class TemporalContextBlock(nn.Module):
    """
    multi-branch temporal block:
    - different branches look at different temporal receptive fields
    - summed outputs are mixed with 1x1 conv
    """
    def __init__(self, channels):
        super().__init__()

        self.dw_t1 = nn.Conv2d(
            channels, channels,
            kernel_size=(3, 7),
            padding=(1, 3),
            groups=channels,
            bias=False
        )

        self.dw_t2 = nn.Conv2d(
            channels, channels,
            kernel_size=(3, 9),
            padding=(1, 8),
            dilation=(1, 2),
            groups=channels,
            bias=False
        )

        self.dw_t3 = nn.Conv2d(
            channels, channels,
            kernel_size=(3, 11),
            padding=(1, 20),
            dilation=(1, 4),
            groups=channels,
            bias=False
        )

        self.norm = make_norm(channels)
        self.mix = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.mix_norm = make_norm(channels)

    def forward(self, x):
        identity = x

        out = self.dw_t1(x) + self.dw_t2(x) + self.dw_t3(x)
        out = self.norm(out)
        out = F.silu(out)

        out = self.mix(out)
        out = self.mix_norm(out)

        out = out + identity
        return F.silu(out)


# main model class
class CNNAutoencoder(nn.Module):
    """
    CNN autoencoder for spectrogram inpainting
    """
    def __init__(self, base_channels=24, predict_residual=True):
        super().__init__()
        self.predict_residual = predict_residual

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stem = nn.Sequential(
            nn.Conv2d(2, c1, kernel_size=3, padding=1, bias=False),
            make_norm(c1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            make_norm(c1),
            nn.SiLU(),
        )

        self.enc1 = LightConvBlock(c1, c1)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)

        self.enc2 = MBConvBlock(c2, c2, expansion=2, kernel_size=3, se_reduction=8)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)

        self.enc3 = MBConvBlock(c3, c3, expansion=3, kernel_size=5, se_reduction=8)
        self.down3 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False)

        self.bottleneck = nn.Sequential(
            MBConvBlock(c4, c4, expansion=2, kernel_size=5, dilation=1, se_reduction=8),
            TemporalContextBlock(c4),
            MBConvBlock(c4, c4, expansion=2, kernel_size=3, dilation=1, se_reduction=8),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = MBConvBlock(c4, c3, expansion=2, kernel_size=5, se_reduction=8)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = MBConvBlock(c3, c2, expansion=2, kernel_size=3, se_reduction=8)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = LightConvBlock(c2, c1)

        self.final_head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            make_norm(c1),
            nn.SiLU(),
            nn.Conv2d(c1, 1, kernel_size=1),
        )

        # training state
        self.history = None
        self.best_score = None
        self.best_epoch = None
        self.loss_name = None
        self.optimiser = None
        self.scheduler = None
        self.scaler = None
        self.device_obj = None
        self.device_type = "cpu"
        self.use_amp = False
        self.scheduler_type = None

    # forward
    def forward(self, x, mask, return_intermediates=False):
        inp = torch.cat([x, mask], dim=1)

        h0 = self.stem(inp)

        h1 = self.enc1(h0)
        h1d = F.silu(self.down1(h1))

        h2 = self.enc2(h1d)
        h2d = F.silu(self.down2(h2))

        h3 = self.enc3(h2d)
        h3d = F.silu(self.down3(h3))

        hb = self.bottleneck(h3d)

        u1 = self.up1(hb)
        d1 = self.dec1(u1)

        u2 = self.up2(d1)
        d2 = self.dec2(u2)

        u3 = self.up3(d2)
        d3 = self.dec3(u3)

        pred = self.final_head(d3)

        # resize back if down/up path created a small mismatch
        if pred.shape[-2:] != x.shape[-2:]:
            pred = F.interpolate(pred, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if self.predict_residual:
            out = x + mask * pred
        else:
            out = pred

        if return_intermediates:
            return {
                "stem": h0,
                "enc1": h1,
                "enc2": h2,
                "enc3": h3,
                "bottleneck": hb,
                "dec1": d1,
                "dec2": d2,
                "dec3": d3,
                "pred": pred,
                "out": out,
            }

        return out

    # loss selection
    def get_loss(self, variant="short_gap", shortgap_loss="masked_l1_grad", longgap_loss="masked_multires_l1_grad"):
        if variant == "short_gap":
            return shortgap_loss
        elif variant == "long_gap":
            return longgap_loss
        else:
            raise ValueError(f"Invalid gap variant {variant}.")

    # compute loss
    def compute_loss(self, pred, target, mask, loss="masked_l1_grad"):
        if loss == "masked_l1":
            total_loss, gap_loss, context_loss = masked_l1_loss(pred, target, mask)
            loss_dict = {
                "loss": total_loss.item(),
                "gap_loss": gap_loss.item(),
                "context_loss": context_loss.item(),
                "grad_loss": 0.0,
            }

        elif loss == "masked_l1_grad":
            total_loss, gap_loss, context_loss, grad_loss = masked_l1_grad_loss(pred, target, mask)
            loss_dict = {
                "loss": total_loss.item(),
                "gap_loss": gap_loss.item(),
                "context_loss": context_loss.item(),
                "grad_loss": grad_loss.item(),
            }

        elif loss == "masked_huber":
            total_loss, gap_loss, context_loss = masked_huber_loss(pred, target, mask)
            loss_dict = {
                "loss": total_loss.item(),
                "gap_loss": gap_loss.item(),
                "context_loss": context_loss.item(),
                "grad_loss": 0.0,
            }

        elif loss == "masked_multires_l1":
            total_loss, gap_loss, context_loss = masked_multires_l1_loss(pred, target, mask)
            loss_dict = {
                "loss": total_loss.item(),
                "gap_loss": gap_loss.item(),
                "context_loss": context_loss.item(),
                "grad_loss": 0.0,
            }

        else:
            raise ValueError(f"Unknown loss name: {loss}")

        return total_loss, loss_dict

    # metrics
    def compute_metrics(self, pred, target, mask):
        return {
            "gap_mae": masked_mae(pred, target, mask).item(),
            "gap_rmse": masked_rmse(pred, target, mask).item(),
            "full_mae": full_mae(pred, target).item(),
            "full_rmse": full_rmse(pred, target).item(),
            "psnr": psnr(pred, target).item(),
        }

    def make_running_dict(self):
        return {
            "loss": 0.0,
            "gap_loss": 0.0,
            "context_loss": 0.0,
            "grad_loss": 0.0,
            "gap_mae": 0.0,
            "gap_rmse": 0.0,
            "full_mae": 0.0,
            "full_rmse": 0.0,
            "psnr": 0.0,
        }

    def get_lr(self):
        return self.optimiser.param_groups[0]["lr"]

    # compile
    def compile(
        self,
        optimiser,
        lossfn,
        device,
        use_amp=True,
        use_scheduler=True,
        scheduler_type="plateau",
        scheduler_factor=0.5,
        scheduler_patience=3,
        scheduler_min_lr=1e-7,
        cosine_tmax=50,
    ):
        self.optimiser = optimiser
        self.loss_name = lossfn

        self.device_obj = device if isinstance(device, torch.device) else torch.device(device)
        self.device_type = self.device_obj.type
        self.to(self.device_obj)

        # AMP only on CUDA
        self.use_amp = use_amp and (self.device_type == "cuda")
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_amp)

        self.scheduler = None
        if use_scheduler:
            if scheduler_type == "plateau":
                self.scheduler = ReduceLROnPlateau(
                    self.optimiser,
                    mode="min",
                    factor=scheduler_factor,
                    patience=scheduler_patience,
                    min_lr=scheduler_min_lr,
                )
            elif scheduler_type == "cosine":
                self.scheduler = CosineAnnealingLR(
                    self.optimiser,
                    T_max=cosine_tmax,
                    eta_min=scheduler_min_lr,
                )
            else:
                raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

        self.scheduler_type = scheduler_type

    # one training epoch
    def train_epoch(self, dataloader, device, grad_clip=1.0):
        self.train()
        running = self.make_running_dict()
        num_batches = 0

        for batch in tqdm(dataloader, leave=False, desc="CNN_AE Train"):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            self.optimiser.zero_grad(set_to_none=True)

            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                pred = self(x, mask)
                total_loss, loss_dict = self.compute_loss(pred, y, mask, loss=self.loss_name)

            self.scaler.scale(total_loss).backward()

            if grad_clip is not None:
                self.scaler.unscale_(self.optimiser)
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

            self.scaler.step(self.optimiser)
            self.scaler.update()

            with torch.no_grad():
                eval_dict = self.compute_metrics(pred.float(), y.float(), mask.float())

            for key in running:
                if key in loss_dict:
                    running[key] += float(loss_dict[key])
                if key in eval_dict:
                    running[key] += float(eval_dict[key])

            num_batches += 1

        return {k: v / max(num_batches, 1) for k, v in running.items()}

    # one validation epoch
    @torch.no_grad()
    def evaluate(self, dataloader, device):
        self.eval()
        running = self.make_running_dict()
        num_batches = 0

        for batch in tqdm(dataloader, leave=False, desc="CNN_AE Val"):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                pred = self(x, mask)
                total_loss, loss_dict = self.compute_loss(pred, y, mask, loss=self.loss_name)

            eval_dict = self.compute_metrics(pred.float(), y.float(), mask.float())

            for key in running:
                if key in loss_dict:
                    running[key] += float(loss_dict[key])
                if key in eval_dict:
                    running[key] += float(eval_dict[key])

            num_batches += 1

        return {k: v / max(num_batches, 1) for k, v in running.items()}

    # fit
    def fit(
        self,
        train_loader,
        val_loader,
        device,
        n_epochs,
        checkpoint_dir,
        history_path,
        monitor="val_gap_rmse",
        mode="min",
        patience=5,
        min_delta=1e-4,
        save_best_after_epoch=5,
        grad_clip=1.0,
        verbose=True,
    ):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)

        self.history = {
            "epoch": [],
            "lr": [],
            "train_loss": [],
            "train_gap_loss": [],
            "train_context_loss": [],
            "train_grad_loss": [],
            "train_gap_mae": [],
            "train_gap_rmse": [],
            "train_full_mae": [],
            "train_full_rmse": [],
            "train_psnr": [],
            "val_loss": [],
            "val_gap_loss": [],
            "val_context_loss": [],
            "val_grad_loss": [],
            "val_gap_mae": [],
            "val_gap_rmse": [],
            "val_full_mae": [],
            "val_full_rmse": [],
            "val_psnr": [],
        }

        
        manager = ModelCheckpoint(
            checkpoint_dir=checkpoint_dir,
            monitor=monitor,
            mode=mode,
            patience=patience,
            min_delta=min_delta,
            save_best_after_epoch=save_best_after_epoch,
            verbose=verbose,
        )

        for epoch in tqdm(range(1, n_epochs + 1), desc="Training CNN-AE"):
            train_metrics = self.train_epoch(train_loader, device=device, grad_clip=grad_clip)
            val_metrics = self.evaluate(val_loader, device=device)

            current_lr = self.get_lr()

            epoch_record = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_gap_loss": train_metrics["gap_loss"],
                "train_context_loss": train_metrics["context_loss"],
                "train_grad_loss": train_metrics["grad_loss"],
                "train_gap_mae": train_metrics["gap_mae"],
                "train_gap_rmse": train_metrics["gap_rmse"],
                "train_full_mae": train_metrics["full_mae"],
                "train_full_rmse": train_metrics["full_rmse"],
                "train_psnr": train_metrics["psnr"],
                "val_loss": val_metrics["loss"],
                "val_gap_loss": val_metrics["gap_loss"],
                "val_context_loss": val_metrics["context_loss"],
                "val_grad_loss": val_metrics["grad_loss"],
                "val_gap_mae": val_metrics["gap_mae"],
                "val_gap_rmse": val_metrics["gap_rmse"],
                "val_full_mae": val_metrics["full_mae"],
                "val_full_rmse": val_metrics["full_rmse"],
                "val_psnr": val_metrics["psnr"],
            }

            # append to history
            for key in self.history:
                self.history[key].append(epoch_record[key])

            pd.DataFrame(self.history).to_csv(history_path, index=False)

            # epoch summary
            if verbose:
                print(f"Epoch {epoch:02d}")
                print(f"Learning Rate:    {current_lr:.8e}")
                print(f"Train Loss:       {train_metrics['loss']:.6f}")
                print(f"Val Loss:         {val_metrics['loss']:.6f}")
                print(f"Train Gap Loss:   {train_metrics['gap_loss']:.6f}")
                print(f"Val Gap Loss:     {val_metrics['gap_loss']:.6f}")
                print(f"Train Gap RMSE:   {train_metrics['gap_rmse']:.6f}")
                print(f"Val Gap RMSE:     {val_metrics['gap_rmse']:.6f}")
                print(f"Train PSNR:       {train_metrics['psnr']:.4f}")
                print(f"Val PSNR:         {val_metrics['psnr']:.4f}")
                print("-" * 60)

            # best and last saving and early stopping to ModelCheckpoint
            manager.step(
                epoch=epoch,
                metrics=epoch_record,
                model=self,
                optimiser=self.optimiser,
            )

            self.best_score = manager.best_score
            self.best_epoch = manager.best_epoch

            # step lr scheduler after validation
            if self.scheduler is not None:
                if self.scheduler_type == "plateau":
                    self.scheduler.step(epoch_record[monitor])
                elif self.scheduler_type == "cosine":
                    self.scheduler.step()

            if manager.should_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        return {
            "history": self.history,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "checkpoint_dir": checkpoint_dir,
            "lossfn": self.loss_name,
        }

    # simple prediction helper
    @torch.no_grad()
    def predict_batch(self, x, mask, device):
        self.eval()
        x = x.to(device)
        mask = mask.to(device)
        return self(x, mask)