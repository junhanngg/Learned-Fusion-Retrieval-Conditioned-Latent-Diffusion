# standard library
import os
import sys
import math
from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
from torchinfo import summary
from huggingface_hub import hf_hub_download

# project root setup
notebook_dir = Path.cwd()
project_root = notebook_dir.parent
sys.path.insert(0, str(project_root))

# local imports
from utils.data_load import DataModule
from utils.checkpoint import ModelCheckpoint
from utils.logging import save_history
from utils.losses import (
    masked_l1_loss,
    masked_l1_grad_loss,
    masked_huber_loss,
    masked_multires_l1_loss,
    masked_mae,
    masked_rmse,
    full_mae,
    full_rmse,
    psnr,
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

--------------------------------------------------------------------------
- Load the Best Model:

model.load_best(checkpoint_dir=checkpoint_dir / "best_model.pt", device=device)

"""

# ===================================================================================================
# helper for groupnorm
def make_norm(channels, max_groups=8):
    """
    create a GroupNorm layer with a number of groups
    """
    groups = min(max_groups, channels)

    # reduce number of groups until channels
    while channels % groups != 0 and groups > 1:
        groups -= 1

    return nn.GroupNorm(groups, channels)

# se block
class SEBlock(nn.Module):
    """
    - learn which channels are important
    - reweight channels adaptively based on global context

    1. squeeze: global average pool each channel to 1 number
    2. excitation: pass through a tiny bottleneck MLP implemented with 1x1 convs
    3. scale: multiply original feature map by learned channel weights
    """
    def __init__(self, channels, reduction=8):
        super().__init__()

        # hidden channel count for the small bottleneck network
        hidden = max(channels // reduction, 8)

        # global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # reduce channel dimension
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)

        # expand back to original channel dimension
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        # compute one summary value per channel
        scale = self.pool(x)

        # small nonlinear bottleneck
        scale = F.silu(self.fc1(scale))

        # output channel weights in [0, 1]
        scale = torch.sigmoid(self.fc2(scale))

        # reweight channels
        return x * scale


# MBConv block
class MBConvBlock(nn.Module):
    """
    - expand channels with 1x1 conv
    - apply depthwise convolution for cheap spatial filtering
    - apply SE attention
    - project back to desired output channels
    - use residual connection when input and output shapes match
    """
    def __init__(self, in_channels, out_channels, expansion=4, kernel_size=3, dilation=1, se_reduction=8):
        super().__init__()

        # only allow identity skip when channel count is unchanged
        self.use_residual = (in_channels == out_channels)
        self.use_expand = expansion > 1

        # hidden channel count inside the block
        hidden_dim = in_channels * expansion if self.use_expand else in_channels

        if self.use_expand:
            # 1x1 expansion convolution increases representation capacity
            self.expand = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
            self.norm1 = make_norm(hidden_dim)

        # padding chosen so spatial size is preserved
        padding = (kernel_size // 2) * dilation

        # depthwise convolution: groups=hidden_dim means each channel is convolved independently
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

        # channel attention
        self.se = SEBlock(hidden_dim, reduction=se_reduction)

        # 1x1 projection back to out_channels
        self.project = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.norm3 = make_norm(out_channels)

    def forward(self, x):
        # start from the input
        out = x

        # optional expansion phase
        if self.use_expand:
            out = self.expand(x)
            out = F.silu(self.norm1(out))

        # cheap channel-wise spatial convolution
        out = self.depthwise(out)
        out = F.silu(self.norm2(out))

        # reweight channels by importance
        out = self.se(out)

        # compress to desired output channels
        out = self.project(out)
        out = self.norm3(out)

        if self.use_residual:
            out = out + x
        return F.silu(out)

# lightweight residual convolution block
class LightConvBlock(nn.Module):
    """
    - use a simpler residual block in high-resolution stages
    - cheaper than MBConv while still expressive enough early in the network
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if channel count matches, use identity residual
        self.use_residual = (in_channels == out_channels)

        # main two-layer convolution path
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

        # main path
        out = self.conv1(x)
        out = F.silu(self.norm1(out))
        out = self.conv2(out)
        out = self.norm2(out)

        # residual addition
        if self.use_residual:
            out = out + identity
        else:
            out = out + self.skip(identity)
        return F.silu(out)


# temporal context block for longer missing gaps
class TemporalContextBlock(nn.Module):
    """
    - use several depthwise convolutions with different temporal receptive fields
    - each branch looks at a different temporal scale
    - combine them and mix across channels with a 1x1 conv
    """
    def __init__(self, channels):
        super().__init__()

        # local temporal context
        self.dw_t1 = nn.Conv2d(
            channels, channels,
            kernel_size=(3, 7),
            padding=(1, 3),
            groups=channels,
            bias=False
        )

        # medium temporal context
        self.dw_t2 = nn.Conv2d(
            channels, channels,
            kernel_size=(3, 9),
            padding=(1, 8),
            dilation=(1, 2),
            groups=channels,
            bias=False
        )

        # larger temporal context
        self.dw_t3 = nn.Conv2d(
            channels, channels,
            kernel_size=(3, 11),
            padding=(1, 20),
            dilation=(1, 4),
            groups=channels,
            bias=False
        )

        # normalise summed branches
        self.norm = make_norm(channels)

        # mix information across channels after the depthwise branches
        self.mix = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.mix_norm = make_norm(channels)

    def forward(self, x):
        # residual connection
        identity = x

        # sum outputs from multiple temporal scales
        out = self.dw_t1(x) + self.dw_t2(x) + self.dw_t3(x)
        out = self.norm(out)
        out = F.silu(out)

        # channel mixing
        out = self.mix(out)
        out = self.mix_norm(out)

        # residual add
        out = out + identity
        return F.silu(out)

# main model class
class CNNAutoencoder(nn.Module):
    """
    - input = masked spectrogram + mask
    - encoder compresses representation while increasing channels
    - bottleneck captures broader structure and temporal context
    - decoder upsamples back to original resolution
    - final output predicts either:
        1. a residual to add inside the masked region, or
        2. a direct reconstruction

    this module also stores training utilities:
        - loss selection
        - metric computation
        - compile
        - fit
        - evaluate
        - checkpoint save/load
    """
    def __init__(self, base_channels=24, predict_residual=True):
        super().__init__()
        self.predict_residual = predict_residual

        # channel widths at each scale
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # stem: takes 2-channel input [masked spectrogram, mask] and maps it into the first feature space
        self.stem = nn.Sequential(
            nn.Conv2d(2, c1, kernel_size=3, padding=1, bias=False),
            make_norm(c1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            make_norm(c1),
            nn.SiLU(),
        )

        # encoder stage 1 at high resolution
        self.enc1 = LightConvBlock(c1, c1)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)

        # encoder stage 2
        self.enc2 = MBConvBlock(c2, c2, expansion=2, kernel_size=3, se_reduction=8)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)

        # encoder stage 3
        self.enc3 = MBConvBlock(c3, c3, expansion=3, kernel_size=5, se_reduction=8)
        self.down3 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False)

        # bottleneck: captures compressed global structure and longer temporal context
        self.bottleneck = nn.Sequential(
            MBConvBlock(c4, c4, expansion=2, kernel_size=5, dilation=1, se_reduction=8),
            TemporalContextBlock(c4),
            MBConvBlock(c4, c4, expansion=2, kernel_size=3, dilation=1, se_reduction=8),
        )

        # decoder stage 1
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = MBConvBlock(c4, c3, expansion=2, kernel_size=5, se_reduction=8)

        # decoder stage 2
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = MBConvBlock(c3, c2, expansion=2, kernel_size=3, se_reduction=8)

        # decoder stage 3
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = LightConvBlock(c2, c1)

        # final head maps decoder features to 1 spectrogram channel
        self.final_head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            make_norm(c1),
            nn.SiLU(),
            nn.Conv2d(c1, 1, kernel_size=1),
        )

        # attributes used during training / checkpointing
        self.history = None
        self.best_score = None
        self.best_epoch = None
        self.loss_name = None
        self.optimiser = None
        self.scheduler = None
        self.scaler = None
        self.device_str = None

    # forward
    def forward(self, x, mask, return_intermediates=False):
        # concatenate masked spectrogram and mask as 2-channel input
        inp = torch.cat([x, mask], dim=1)

        # initial feature extraction
        h0 = self.stem(inp)

        # encoder stage 1
        h1 = self.enc1(h0)
        h1d = F.silu(self.down1(h1))

        # encoder stage 2
        h2 = self.enc2(h1d)
        h2d = F.silu(self.down2(h2))

        # encoder stage 3
        h3 = self.enc3(h2d)
        h3d = F.silu(self.down3(h3))

        # bottleneck representation
        hb = self.bottleneck(h3d)

        # decoder stage 1
        u1 = self.up1(hb)
        d1 = self.dec1(u1)

        # decoder stage 2
        u2 = self.up2(d1)
        d2 = self.dec2(u2)

        # decoder stage 3
        u3 = self.up3(d2)
        d3 = self.dec3(u3)

        # raw prediction from decoder features
        pred = self.final_head(d3)

        # if size changed due to down/up sampling mismatch, resize back to input size
        if pred.shape[-2:] != x.shape[-2:]:
            pred = F.interpolate(pred, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if self.predict_residual:
            out = x + mask * pred
        else:
            out = pred

        # debugging / visualisation output
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
        """
        choose the training loss name based on the gap regime.
        """
        if variant == "short_gap":
            return shortgap_loss
        elif variant == "long_gap":
            return longgap_loss
        else:
            raise ValueError(f"Invalid gap variant {variant}.")

    # compute loss
    def compute_loss(self, pred, target, mask, loss="masked_l1_grad"):
        """
        compute the chosen loss and return:
        - total differentiable loss tensor
        - a logging dictionary of scalar components
        """
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

    # helpers
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

    # return current lr from the first parameter group
    def get_lr(self):
        return self.optimiser.param_groups[0]["lr"]

    # store training configs on the model 
    def compile(self, optimiser, lossfn, device, use_amp=True, 
                use_scheduler=True, scheduler_type="plateau", scheduler_factor=0.5,
                scheduler_patience=3, scheduler_min_lr=1e-7, cosine_tmax=50
                ):
        self.optimiser = optimiser
        self.loss_name = lossfn
        self.device_str = str(device)

        # move model weights to the requested device
        self.to(device)

        amp_enabled = use_amp and ("cuda" in self.device_str)
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        # build scheduler 
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
        self.use_amp = amp_enabled

    # one training epoch
    def train_epoch(self, dataloader, device, grad_clip=1.0):
        """
        train for one full epoch over the dataloader
        """
        self.train()
        running = self.make_running_dict()
        num_batches = 0

        for batch in tqdm(dataloader, leave=False, desc="CNN_AE Train"):
            # move tensors to target device
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            # clear old gradients
            self.optimiser.zero_grad(set_to_none=True)

            # mixed precision forward pass
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred = self(x, mask)
                total_loss, loss_dict = self.compute_loss(pred, y, mask, loss=self.loss_name)

            # scaled backward pass
            self.scaler.scale(total_loss).backward()

            # unscale before clipping
            if grad_clip is not None:
                self.scaler.unscale_(self.optimiser)
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

            # optimizer step
            self.scaler.step(self.optimiser)
            self.scaler.update()

            # compute metrics in full precision for stable logging
            with torch.no_grad():
                eval_dict = self.compute_metrics(pred.float(), y.float(), mask.float())

            # accumulate batch losses and metrics
            for key in running:
                if key in loss_dict:
                    running[key] += float(loss_dict[key])
                if key in eval_dict:
                    running[key] += float(eval_dict[key])

            num_batches += 1

        # average over batches
        return {k: v / max(num_batches, 1) for k, v in running.items()}

    # one validation epoch
    @torch.no_grad()
    def evaluate(self, dataloader, device):
        """
        evaluate model over one full dataloader without updating parameters
        """
        self.eval()
        running = self.make_running_dict()
        num_batches = 0

        for batch in tqdm(dataloader, leave=False, desc="CNN_AE Val"):
            # move tensors to device
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            # forward pass only
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred = self(x, mask)
                total_loss, loss_dict = self.compute_loss(pred, y, mask, loss=self.loss_name)

            # compute eval metrics
            eval_dict = self.compute_metrics(pred.float(), y.float(), mask.float())

            # accumulate all metrics
            for key in running:
                if key in loss_dict:
                    running[key] += float(loss_dict[key])
                if key in eval_dict:
                    running[key] += float(eval_dict[key])

            num_batches += 1

        # average over batches
        return {k: v / max(num_batches, 1) for k, v in running.items()}

    # checkpoint save
    def save_checkpoint(self, path, epoch, best_score=None, best_epoch=None):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict() if self.optimiser is not None else None,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "history": self.history,
            "lossfn": self.loss_name,
        }
        torch.save(checkpoint, path)

    # checkpoint load
    def load_checkpoint(self, path, device):
        checkpoint = torch.load(path, map_location=device)

        # restore model weights
        self.load_state_dict(checkpoint["model_state_dict"])

        # restore optimizer state if possible
        if self.optimiser is not None and checkpoint.get("optimiser_state_dict") is not None:
            self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

        # restore stored metadata
        self.history = checkpoint.get("history", None)
        self.best_score = checkpoint.get("best_score", None)
        self.best_epoch = checkpoint.get("best_epoch", None)
        self.loss_name = checkpoint.get("lossfn", self.loss_name)

        return checkpoint

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
        grad_clip=1.0
    ):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)

        # initialise full history structure
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

        # initialise best score depending on monitor direction
        best_score = float("inf") if mode == "min" else -float("inf")
        best_epoch = None
        bad_epochs = 0

        # epoch loop
        for epoch in tqdm(range(1, n_epochs + 1), desc="Training CNN-AE"):
            # one training pass
            train_metrics = self.train_epoch(train_loader, device=device, grad_clip=grad_clip)

            # one validation pass
            val_metrics = self.evaluate(val_loader, device=device)

            # current optimiser learning rate
            current_lr = self.get_lr()

            # pack everything from this epoch into one dict
            record = {
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

            # append record values into history lists
            for key in self.history:
                self.history[key].append(record[key])

            # save history to CSV after every epoch
            pd.DataFrame(self.history).to_csv(history_path, index=False)

            # metric used for best model / early stopping
            current_score = record[monitor]
            improved = False

            # improvement test depends on whether smaller or larger is better
            if mode == "min":
                is_better = current_score < (best_score - min_delta)
            else:
                is_better = current_score > (best_score + min_delta)

            # update best model tracking only after a warmup period
            if epoch >= save_best_after_epoch and is_better:
                best_score = current_score
                best_epoch = epoch
                bad_epochs = 0
                improved = True
            elif epoch >= save_best_after_epoch:
                bad_epochs += 1

            # store best info on the model object too
            self.best_score = best_score
            self.best_epoch = best_epoch

            # always save the latest checkpoint
            self.save_checkpoint(
                checkpoint_dir / "last_model.pt",
                epoch=epoch,
                best_score=best_score,
                best_epoch=best_epoch,
            )

            # save separate best checkpoint when improved
            if improved:
                self.save_checkpoint(
                    checkpoint_dir / "best_model.pt",
                    epoch=epoch,
                    best_score=best_score,
                    best_epoch=best_epoch,
                )

            # step lr scheduler
            if self.scheduler is not None:
                if self.scheduler_type == "plateau":
                    self.scheduler.step(current_score)
                elif self.scheduler_type == "cosine":
                    self.scheduler.step()

            # epoch summary printing
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
            if improved:
                print(f"[Checkpoint] New best model saved at epoch {epoch} | {monitor} = {current_score:.6f}")
            print("-" * 60)

            # early stopping
            if epoch >= save_best_after_epoch and bad_epochs >= patience:
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