# import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import math
from pathlib import Path

# .py scripts imports
from utils.checkpoint import ModelCheckpoint
from utils.logging import save_history
from utils.losses import (masked_l1_loss, # for 0.5 and 2.0 second 
                          masked_l1_grad_loss,
                          masked_huber_loss,
                          masked_multires_l1_loss, # for 3.0 and 5.0 second gaps

                          # evaluation metrics
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

model = CNNAutoencoder(
    base_channel=16,                        # 24 if there is memory
    shortgap_loss="masked_l1",              # for 0.5 and 2.0
    longgap_loss="masked_multires_l1"       # for 3.0 and 5.0
).to(device)

--------------------------------------------------------------------------
- Creating optimiser:

optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

--------------------------------------------------------------------------
- Get Loss Name:

lossfn = model.get_loss_name(
                        variant="short_gap,
                        shortgap_loss="masked_l1",
                        longgap_loss="masked_multires_l1_grad"
                        )
--------------------------------------------------------------------------
- Training model:

results = model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    optimiser=optimiser,
    device=device,
    n_epochs=10,
    checkpoint_dir=checkpoint_dir,
    history_csv_path=history_dir,
    loss_name=lossfn,
    monitor="val_gap_rmse",
    mode="min",
    patience=5,
    min_delta=1e-4,
    save_best_after_epoch=5,
    grad_clip=1.0,
    use_amp=True,
)
--------------------------------------------------------------------------
- Load the Best Model:

model.load_best(checkpoint_dir=checkpoint_dir, device=device)

--------------------------------------------------------------------------
- Inferencing:

batch = next(iter(test_loader))

x = batch["x"][0]
mask = batch["mask"][0]

pred = model.predict(x, mask, device=device)
print(pred.shape)

pred, merged = model.predict_full(x, mask, device=device) # full mmerged reconstruction

"""

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# define checkpoints and history 
root_dir = Path("training_logs/cnn_ae")
# root_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir = root_dir / "checkpoints"
history_dir = root_dir / "history"

# SE block
# learns channel-wise importance weights
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        # hidden size inside the gating MLP
        hidden = max(channels // reduction, 8)

        # global average pooling reduces (H, W) -> 1 value per channel
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 1x1 conv acts like a tiny fully connected layer over channels
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        # squeeze spatial dimensions
        scale = self.pool(x)

        # small nonlinear channel transform
        scale = F.silu(self.fc1(scale))

        # convert to channel weights in [0, 1]
        scale = torch.sigmoid(self.fc2(scale))

        # reweight each channel
        return x * scale


# mobile inverted bottleneck convolution block
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, kernel_size=3, se_reduction=8):
        super().__init__()

        # residual only if channel size stays the same
        self.use_residual = (in_channels == out_channels)

        # whether to expand channels before depthwise conv
        self.use_expand = expansion > 1
        hidden_dim = in_channels * expansion if self.use_expand else in_channels

        # optional expansion layer
        if self.use_expand:
            self.expand = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)

        # depthwise convolution: one filter per channel
        self.depthwise = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        # channel attention
        self.se = SEBlock(hidden_dim, reduction=se_reduction)

        # projection back to desired output channels
        self.project = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # IMPORTANT: start with x so out is always defined
        out = x

        # optional expansion
        if self.use_expand:
            out = self.expand(out)
            out = F.silu(self.bn1(out))

        # depthwise spatial filtering
        out = self.depthwise(out)
        out = F.silu(self.bn2(out))

        # channel attention
        out = self.se(out)

        # projection to output size
        out = self.project(out)
        out = self.bn3(out)

        # residual if dimensions match
        if self.use_residual:
            out = out + x

        return F.silu(out)


# lightweight residual conv block
class LightConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.use_residual = (in_channels == out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # projection skip if channel sizes differ
        if not self.use_residual:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = F.silu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_residual:
            out = out + identity
        else:
            out = out + self.skip(identity)

        return F.silu(out)

class CNNAutoencoder(nn.Module):
    """
    Main CNN Autoencoder class

    Includes:
    - architecture
    - training
    - validation
    - checkpoint loading
    - inference
    - visual inspection
    """
    def __init__(
        self,
        base_channels=24,
        shortgap_loss="masked_l1_grad",
        longgap_loss="masked_multires_l1_grad"
    ):
        super().__init__()

        # store default loss names for convenience
        self.shortgap_loss = shortgap_loss
        self.longgap_loss = longgap_loss

        # channel sizes across network stages
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # stem concatenated into 2 channels
        self.stem = nn.Sequential(
            nn.Conv2d(2, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )

        # encoder
        self.enc1 = LightConvBlock(c1, c1)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)

        self.enc2 = MBConvBlock(c2, c2, expansion=2, kernel_size=3, se_reduction=8)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)

        self.enc3 = MBConvBlock(c3, c3, expansion=3, kernel_size=5, se_reduction=8)
        self.down3 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False)

        # bottleneck
        self.bottleneck = nn.Sequential(
            MBConvBlock(c4, c4, expansion=3, kernel_size=5, se_reduction=8),
            MBConvBlock(c4, c4, expansion=3, kernel_size=3, se_reduction=8)
        )

        # decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = MBConvBlock(c4, c3, expansion=2, kernel_size=5, se_reduction=8)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = MBConvBlock(c3, c2, expansion=2, kernel_size=3, se_reduction=8)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = LightConvBlock(c2, c1)

        # final single-channel reconstruction
        self.final_conv = nn.Conv2d(c1, 1, kernel_size=1)

    # forward pass
    def forward(self, x, mask, return_intermediates=False):
        # concatenate masked spectrogram and binary mask
        inp = torch.cat([x, mask], dim=1)

        # stem
        h0 = self.stem(inp)

        # encoder
        h1 = self.enc1(h0)
        h1d = F.silu(self.down1(h1))

        h2 = self.enc2(h1d)
        h2d = F.silu(self.down2(h2))

        h3 = self.enc3(h2d)
        h3d = F.silu(self.down3(h3))

        # bottleneck
        hb = self.bottleneck(h3d)

        # decoder
        u1 = self.up1(hb)
        d1 = self.dec1(u1)

        u2 = self.up2(d1)
        d2 = self.dec2(u2)

        u3 = self.up3(d2)
        d3 = self.dec3(u3)

        # map back to 1-channel spectrogram
        out = self.final_conv(d3)

        # safety resize if spatial dimensions mismatch
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

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
                "out": out,
            }

        return out

    # loss selection helper
    def get_loss_name(self, variant="short_gap", shortgap_loss=None, longgap_loss=None):
        # allow override during call, otherwise use defaults stored in the model
        shortgap_loss = shortgap_loss or self.shortgap_loss
        longgap_loss = longgap_loss or self.longgap_loss

        if variant == "short_gap":
            return shortgap_loss
        elif variant == "long_gap":
            return longgap_loss
        else:
            raise ValueError(f"Invalid gap variant: {variant}")

    # compute training loss
    def compute_loss(self, pred, target, mask, loss_name="masked_l1_grad"):
        if loss_name == "masked_l1":
            total_loss, gap_loss, context_loss = masked_l1_loss(pred, target, mask)
            log_dict = {
                "loss": float(total_loss.item()),
                "gap_loss": float(gap_loss.item()),
                "context_loss": float(context_loss.item()),
                "grad_loss": 0.0,
            }

        elif loss_name == "masked_l1_grad":
            total_loss, gap_loss, context_loss, grad_loss = masked_l1_grad_loss(pred, target, mask)
            log_dict = {
                "loss": float(total_loss.item()),
                "gap_loss": float(gap_loss.item()),
                "context_loss": float(context_loss.item()),
                "grad_loss": float(grad_loss.item()),
            }

        elif loss_name == "masked_huber":
            total_loss, gap_loss, context_loss = masked_huber_loss(pred, target, mask)
            log_dict = {
                "loss": float(total_loss.item()),
                "gap_loss": float(gap_loss.item()),
                "context_loss": float(context_loss.item()),
                "grad_loss": 0.0,
            }

        elif loss_name == "masked_multires_l1":
            total_loss, gap_loss, context_loss = masked_multires_l1_loss(pred, target, mask)
            log_dict = {
                "loss": float(total_loss.item()),
                "gap_loss": float(gap_loss.item()),
                "context_loss": float(context_loss.item()),
                "grad_loss": 0.0,
            }

        else:
            raise ValueError(f"Unknown loss_name: {loss_name}")

        return total_loss, log_dict

    # compute evaluation metrics
    def compute_metrics(self, pred, target, mask):
        return {
            "gap_mae": float(masked_mae(pred, target, mask).item()),
            "gap_rmse": float(masked_rmse(pred, target, mask).item()),
            "full_mae": float(full_mae(pred, target).item()),
            "full_rmse": float(full_rmse(pred, target).item()),
            "psnr": float(psnr(pred, target).item()),
        }

    # train for one epoch
    def train_epoch(self, dataloader, optimiser, device, loss_name, grad_clip=1.0, use_amp=True):
        # standard PyTorch train mode
        super().train()

        running = {
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

        num_batches = 0

        # AMP only on CUDA
        use_amp = use_amp and ("cuda" in str(device))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for batch in tqdm(dataloader, leave=False):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = self(x, mask)
                total_loss, loss_dict = self.compute_loss(pred, y, mask, loss_name=loss_name)

            scaler.scale(total_loss).backward()

            if grad_clip is not None:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

            scaler.step(optimiser)
            scaler.update()

            with torch.no_grad():
                eval_dict = self.compute_metrics(pred.float(), y.float(), mask.float())

            for key, value in loss_dict.items():
                running[key] += value

            for key, value in eval_dict.items():
                running[key] += value

            num_batches += 1

        return {k: v / max(num_batches, 1) for k, v in running.items()}

    # evaluate for one epoch
    @torch.no_grad()
    def eval_epoch(self, dataloader, device, loss_name, use_amp=True):
        # standard PyTorch eval mode
        super().eval()

        running = {
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

        num_batches = 0
        use_amp = use_amp and ("cuda" in str(device))

        for batch in tqdm(dataloader, desc="Val", leave=False):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = self(x, mask)
                total_loss, loss_dict = self.compute_loss(pred, y, mask, loss_name=loss_name)

            eval_dict = self.compute_metrics(pred.float(), y.float(), mask.float())

            for key, value in loss_dict.items():
                running[key] += value

            for key, value in eval_dict.items():
                running[key] += value

            num_batches += 1

        return {k: v / max(num_batches, 1) for k, v in running.items()}

    # full training loop
    def fit(
        self,
        train_loader,
        val_loader,
        optimiser,
        device,
        n_epochs,
        checkpoint_dir,
        history_csv_path,
        loss_name=None,
        monitor="val_gap_rmse",
        mode="min",
        patience=5,
        min_delta=1e-4,
        save_best_after_epoch=5,
        grad_clip=1.0,
        use_amp=True,
    ):
        # create checkpoint directory if needed
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history = {
            "epoch": [],
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

        # resolve loss name if user did not pass one
        if loss_name is None:
            loss_name = self.shortgap_loss

        # your external checkpoint manager
        manager = ModelCheckpoint(
            checkpoint_dir=checkpoint_dir,
            monitor=monitor,
            mode=mode,
            patience=patience,
            min_delta=min_delta,
            save_best_after_epoch=save_best_after_epoch,
            verbose=True,
        )

        for epoch in tqdm(range(1, n_epochs + 1), desc="Training"):
            train_metrics = self.train_epoch(
                dataloader=train_loader,
                optimiser=optimiser,
                device=device,
                loss_name=loss_name,
                grad_clip=grad_clip,
                use_amp=use_amp,
            )

            val_metrics = self.eval_epoch(
                dataloader=val_loader,
                device=device,
                loss_name=loss_name,
                use_amp=use_amp,
            )

            epoch_record = {
                "epoch": epoch,
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

            for key in history:
                history[key].append(epoch_record[key])

            print(f"Epoch {epoch:02d}")
            print(f"Train Loss:     {train_metrics['loss']:.6f}")
            print(f"Val Loss:       {val_metrics['loss']:.6f}")
            print(f"Train Gap RMSE: {train_metrics['gap_rmse']:.6f}")
            print(f"Val Gap RMSE:   {val_metrics['gap_rmse']:.6f}")
            print(f"Train PSNR:     {train_metrics['psnr']:.4f}")
            print(f"Val PSNR:       {val_metrics['psnr']:.4f}")
            print("-" * 60)

            manager.step(
                epoch=epoch,
                metrics=epoch_record,
                model=self,
                optimiser=optimiser,
            )

            if manager.should_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # save history csv with your own utility
        save_history(history, history_csv_path)

        return {
            "history": history,
            "best_score": manager.best_score,
            "best_epoch": manager.best_epoch,
            "checkpoint_dir": checkpoint_dir,
            "loss_name": loss_name,
        }

    # load checkpoint from a path
    def load_checkpoint(self, checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded model from epoch:", checkpoint.get("epoch", "unknown"))
        print("Best monitored score:", checkpoint.get("best_score", "unknown"))
        return checkpoint

    # convenience method for loading best_model.pt
    def load_best(self, checkpoint_dir, device):
        checkpoint_dir = Path(checkpoint_dir)
        best_model_path = checkpoint_dir / "best_model.pt"
        return self.load_checkpoint(best_model_path, device)

    # predict raw reconstruction
    @torch.no_grad()
    def predict(self, x, mask, device):
        super().eval()

        # if single sample is (C, H, W), add batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)

        x = x.to(device)
        mask = mask.to(device)

        pred = self(x, mask)
        return pred

    # merge model prediction into masked region only
    # keep known context unchanged
    @staticmethod
    def reconstruct(masked_input, pred, mask):
        return masked_input * (1.0 - mask) + pred * mask

    # predict and directly return merged final spectrogram
    @torch.no_grad()
    def predict_full(self, x, mask, device):
        pred = self.predict(x, mask, device)
        x = x.to(device)
        mask = mask.to(device)

        if x.dim() == 3:
            x = x.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)

        merged = self.reconstruct(x, pred, mask)
        return pred, merged

    # visual inspection of results on a dataloader batch
    @torch.no_grad()
    def inspect_results(self, dataloader, device, n_examples=2, show_mask=True):
        super().eval()

        batch = next(iter(dataloader))

        x = batch["x"][:n_examples].to(device)
        y = batch["y"][:n_examples].to(device)
        m = batch["mask"][:n_examples].to(device)

        pred = self(x, m)
        merged = self.reconstruct(x, pred, m)

        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        pred_np = pred.cpu().numpy()
        merged_np = merged.cpu().numpy()
        mask_np = m.cpu().numpy()

        n_cols = 5 if show_mask else 4
        plt.figure(figsize=(4.5 * n_cols, 4 * n_examples))

        for i in range(min(n_examples, x_np.shape[0])):
            col = 1

            plt.subplot(n_examples, n_cols, n_cols * i + col)
            plt.imshow(x_np[i, 0], aspect="auto", origin="lower")
            plt.title(f"Masked input {i}")
            plt.colorbar()
            col += 1

            plt.subplot(n_examples, n_cols, n_cols * i + col)
            plt.imshow(y_np[i, 0], aspect="auto", origin="lower")
            plt.title(f"Ground truth {i}")
            plt.colorbar()
            col += 1

            plt.subplot(n_examples, n_cols, n_cols * i + col)
            plt.imshow(pred_np[i, 0], aspect="auto", origin="lower")
            plt.title(f"Raw recon {i}")
            plt.colorbar()
            col += 1

            plt.subplot(n_examples, n_cols, n_cols * i + col)
            plt.imshow(merged_np[i, 0], aspect="auto", origin="lower")
            plt.title(f"Final recon {i}")
            plt.colorbar()
            col += 1

            if show_mask:
                plt.subplot(n_examples, n_cols, n_cols * i + col)
                plt.imshow(mask_np[i, 0], aspect="auto", origin="lower")
                plt.title(f"Mask {i}")
                plt.colorbar()

        plt.tight_layout()
        plt.show()

    # visual inspection of intermediate feature maps
    @torch.no_grad()
    def inspect_stages(self, dataloader, device, n_examples=1):
        super().eval()

        batch = next(iter(dataloader))

        x = batch["x"][:n_examples].to(device)
        y = batch["y"][:n_examples].to(device)
        m = batch["mask"][:n_examples].to(device)

        stage_dict = self(x, m, return_intermediates=True)
        stage_names = ["stem", "enc1", "enc2", "enc3", "bottleneck", "dec1", "dec2", "dec3", "out"]

        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()

        plt.figure(figsize=(4 * (2 + len(stage_names)), 4 * n_examples))

        for i in range(n_examples):
            plt.subplot(n_examples, 2 + len(stage_names), i * (2 + len(stage_names)) + 1)
            plt.imshow(x_np[i, 0], aspect="auto", origin="lower")
            plt.title("Masked input")
            plt.colorbar()

            plt.subplot(n_examples, 2 + len(stage_names), i * (2 + len(stage_names)) + 2)
            plt.imshow(y_np[i, 0], aspect="auto", origin="lower")
            plt.title("Ground truth")
            plt.colorbar()

            for j, name in enumerate(stage_names):
                feat = stage_dict[name][i, 0].detach().cpu().numpy()
                plt.subplot(n_examples, 2 + len(stage_names), i * (2 + len(stage_names)) + 3 + j)
                plt.imshow(feat, aspect="auto", origin="lower")
                plt.title(name)
                plt.colorbar()

        plt.tight_layout()
        plt.show()