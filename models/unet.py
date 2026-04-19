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
    masked_huber_grad_loss,
    masked_multires_l1_loss,
    masked_multires_l1_grad_loss,
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
model = UNet(n_channels=2, n_classes=1).to(device)
--------------------------------------------------------------------------
- Get Loss function:
lossfn = model.get_loss(
    variant="long_gap",
    shortgap_loss="masked_l1_grad",
    longgap_loss="masked_multires_l1"
)
print(lossfn)
--------------------------------------------------------------------------
- Build optimiser:
optimiser = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
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
    scheduler_min_lr=1e-6,
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
    patience=8,
    min_delta=1e-4,
    save_best_after_epoch=2,
    grad_clip=1.0,
    context_weight_decay=True,
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
    def __init__(self, channels, reduction=4):
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
    - 1x1 expansion to wider hidden dimension
    - depthwise convolution
    - SE attention
    - 1x1 projection back to target width
    - residual connection when shapes match
    """
    def __init__(self, in_channels, out_channels, expansion=4, kernel_size=3):
        super().__init__()

        hidden_dim = in_channels * expansion
        self.use_residual = (in_channels == out_channels)

        # 1x1 expansion
        self.expand = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = make_norm(hidden_dim)

        # depthwise spatial conv
        self.depthwise = nn.Conv2d(
            hidden_dim, hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
            bias=False
        )
        self.bn2 = make_norm(hidden_dim)

        # channel attention
        self.se = SEBlock(hidden_dim)

        # 1x1 projection
        self.project = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = make_norm(out_channels)

    def forward(self, x):
        out = F.silu(self.bn1(self.expand(x)))
        out = F.silu(self.bn2(self.depthwise(out)))
        out = self.se(out)
        out = self.bn3(self.project(out))

        if self.use_residual:
            out = out + x

        return F.silu(out)


# attention gate for skip connections
class AttentionGate(nn.Module):
    """
    attention gate for skip connections:
    - projects decoder gating signal and encoder skip features to shared space
    - learns a per-pixel attention map in [0, 1]
    - suppresses encoder features inside the gap (derived from zeros)
    - passes encoder features from context regions unchanged
    - uses a soft variant with floor alpha so gate ranges in [alpha, 1]
    """
    def __init__(self, F_g, F_l, F_int, alpha=0.7):
        super().__init__()
        self.alpha = alpha

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            make_norm(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            make_norm(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, g, x):
        # align spatial dims if upsampling caused 1-pixel mismatch
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(F.silu(g1 + x1))

        # soft gate: maps [0,1] to [alpha, 1]
        gate = self.alpha + (1.0 - self.alpha) * psi
        return x * gate


# dilated residual block for bottleneck
class DilatedBlock(nn.Module):
    """
    dilated convolution with residual connection:
    - expands receptive field without downsampling
    - a 3x3 kernel with dilation=d has effective size (2d+1) using 9 params
    - stacking rates [2, 4, 8] gives exponentially growing temporal reach
    """
    def __init__(self, channels, dilation=2, kernel_size=3):
        super().__init__()

        padding = dilation * (kernel_size // 2)

        self.conv = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False
        )
        self.bn = make_norm(channels)

    def forward(self, x):
        return F.silu(self.bn(self.conv(x))) + x


# plain residual convolution block
class ResidualBlock(nn.Module):
    """
    lightweight residual block:
    - two 3x3 convolutions with group norm
    - optional 1x1 projection skip when channel counts differ
    - used in bottleneck and decoder where MBConv expansion is too memory-heavy
    """
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.norm1 = make_norm(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.norm2 = make_norm(out_ch)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = F.silu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))

        out = F.silu(out + identity)
        return out


# safe concat for encoder-decoder skip connections
def _safe_cat(decoder_feat, encoder_feat):
    """
    concatenate along channels with spatial alignment for odd-dimension
    edge cases from stride-2 downsampling.
    """
    if decoder_feat.shape[-2:] != encoder_feat.shape[-2:]:
        encoder_feat = F.interpolate(
            encoder_feat,
            size=decoder_feat.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
    return torch.cat([decoder_feat, encoder_feat], dim=1)


# context weight schedule (VIAI-inspired)
def get_context_weight(epoch, initial=0.12, decay_rate=0.9, floor=0.03, decay_every=8):
    """
    Decaying context weight schedule.

    alpha(t) = max(floor, initial * decay_rate^((t-1) / decay_every))

    Early training (epoch 1): context_weight = initial (default 0.12)
    Later training:           decays toward floor (default 0.03)
    """
    return max(floor, initial * (decay_rate ** ((epoch - 1) / decay_every)))


# main model class
class UNet(nn.Module):
    """
    U-Net for spectrogram inpainting.

    Architectural features vs CNN autoencoder:
    - Skip connections with attention gates at each resolution
    - Multi-scale mask injection at every encoder stage
    - Dilated convolution stack in the bottleneck [2, 4, 8]
    - 4 encoder/decoder stages (channels 32 -> 512)

    Architecture:

    Input (B,2,H,W)  <- cat(corrupted_spec, mask)
      |
    Stem [MBConv] ---------------------------------- e0 (32, H, W)
      | strided conv + mask inject                         | attn gate
      v                                                    v
    Enc1 [MBConv] ------------------------ e1 (64, H/2, W/2)
      | strided conv + mask inject                 | attn gate
      v                                            v
    Enc2 [MBConv] ----------------- e2 (128, H/4, W/4)
      | strided conv + mask inject         | attn gate
      v                                    v
    Enc3 [MBConv] ---- e3 (256, H/8, W/8)
      | strided conv        | attn gate
      v                     v
    Bottleneck (512, H/16, W/16)
      | ResidualBlock + dilated [2, 4, 8] + ResidualBlock
      v
    Dec1 <- fuse(up, attn(e3)) [Residual] -> 256
      v
    Dec2 <- fuse(up, attn(e2)) [Residual] -> 128
      v
    Dec3 <- fuse(up, attn(e1)) [Residual] -> 64
      v
    Dec4 <- fuse(up, attn(e0)) [Residual] -> 32
      v
    Output (B, 1, H, W)
    """
    def __init__(self, n_channels=2, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1, bias=False),
            make_norm(32),
            nn.SiLU(),
        )
        self.stem_mb = MBConvBlock(32, 32, expansion=2, kernel_size=3)

        # encoder stage 1
        self.down1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_down1 = make_norm(64)
        self.enc1 = MBConvBlock(64 + 1, 64, expansion=4, kernel_size=3)  # +1 for mask injection

        # encoder stage 2
        self.down2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_down2 = make_norm(128)
        self.enc2 = MBConvBlock(128 + 1, 128, expansion=4, kernel_size=3)

        # encoder stage 3
        self.down3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_down3 = make_norm(256)
        self.enc3 = MBConvBlock(256 + 1, 256, expansion=4, kernel_size=5)

        # encoder stage 4 (into bottleneck)
        self.down4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_down4 = make_norm(512)

        # bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(512, 512, kernel_size=5),
            DilatedBlock(512, dilation=2, kernel_size=3),
            DilatedBlock(512, dilation=4, kernel_size=3),
            DilatedBlock(512, dilation=8, kernel_size=3),
            ResidualBlock(512, 512, kernel_size=3),
        )

        # decoder stage 1
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.attn1 = AttentionGate(F_g=512, F_l=256, F_int=64, alpha=0.7)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(512 + 256, 512, kernel_size=1, bias=False),
            make_norm(512),
            nn.SiLU(),
        )
        self.dec1 = ResidualBlock(512, 256, kernel_size=5)

        # decoder stage 2
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.attn2 = AttentionGate(F_g=256, F_l=128, F_int=32, alpha=0.7)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(256 + 128, 256, kernel_size=1, bias=False),
            make_norm(256),
            nn.SiLU(),
        )
        self.dec2 = ResidualBlock(256, 128, kernel_size=3)

        # decoder stage 3
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.attn3 = AttentionGate(F_g=128, F_l=64, F_int=16, alpha=0.7)
        self.fuse3 = nn.Sequential(
            nn.Conv2d(128 + 64, 128, kernel_size=1, bias=False),
            make_norm(128),
            nn.SiLU(),
        )
        self.dec3 = ResidualBlock(128, 64, kernel_size=3)

        # decoder stage 4
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.attn4 = AttentionGate(F_g=64, F_l=32, F_int=8, alpha=0.7)
        self.fuse4 = nn.Sequential(
            nn.Conv2d(64 + 32, 64, kernel_size=1, bias=False),
            make_norm(64),
            nn.SiLU(),
        )
        self.dec4 = ResidualBlock(64, 32, kernel_size=3)

        # output head
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

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
        # concatenate corrupted spectrogram + mask along channel dimension
        inp = torch.cat([x, mask], dim=1)

        # stem
        e0 = self.stem_mb(self.stem(inp))

        # encoder 1 + mask injection
        h = F.silu(self.bn_down1(self.down1(e0)))
        m1 = F.interpolate(mask, size=h.shape[-2:], mode="nearest")
        e1 = self.enc1(torch.cat([h, m1], dim=1))

        # encoder 2 + mask injection
        h = F.silu(self.bn_down2(self.down2(e1)))
        m2 = F.interpolate(mask, size=h.shape[-2:], mode="nearest")
        e2 = self.enc2(torch.cat([h, m2], dim=1))

        # encoder 3 + mask injection
        h = F.silu(self.bn_down3(self.down3(e2)))
        m3 = F.interpolate(mask, size=h.shape[-2:], mode="nearest")
        e3 = self.enc3(torch.cat([h, m3], dim=1))

        # encoder 4 -> bottleneck
        h = F.silu(self.bn_down4(self.down4(e3)))
        hb = self.bottleneck(h)

        # decoder 1
        h = self.up1(hb)
        e3_att = self.attn1(g=h, x=e3)
        h = _safe_cat(h, e3_att)
        h = self.fuse1(h)
        d1 = self.dec1(h)

        # decoder 2
        h = self.up2(d1)
        e2_att = self.attn2(g=h, x=e2)
        h = _safe_cat(h, e2_att)
        h = self.fuse2(h)
        d2 = self.dec2(h)

        # decoder 3
        h = self.up3(d2)
        e1_att = self.attn3(g=h, x=e1)
        h = _safe_cat(h, e1_att)
        h = self.fuse3(h)
        d3 = self.dec3(h)

        # decoder 4
        h = self.up4(d3)
        e0_att = self.attn4(g=h, x=e0)
        h = _safe_cat(h, e0_att)
        h = self.fuse4(h)
        d4 = self.dec4(h)

        # output
        out = self.outc(d4)

        # safety resize for odd spatial dims
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if return_intermediates:
            return {
                "stem": e0,
                "enc1": e1,
                "enc2": e2,
                "enc3": e3,
                "bottleneck": hb,
                "dec1": d1,
                "dec2": d2,
                "dec3": d3,
                "dec4": d4,
                "out": out,
            }

        return out

    # loss selection
    def get_loss(self, variant="short_gap", shortgap_loss="masked_l1_grad", longgap_loss="masked_multires_l1"):
        if variant == "short_gap":
            return shortgap_loss
        elif variant == "long_gap":
            return longgap_loss
        else:
            raise ValueError(f"Invalid gap variant {variant}.")

    # compute loss
    def compute_loss(self, pred, target, mask, loss="masked_l1_grad", context_weight=0.1):
        if loss == "masked_l1":
            total_loss, gap_loss, context_loss = masked_l1_loss(
                pred, target, mask, context_weight=context_weight
            )
            loss_dict = {
                "loss": total_loss.item(),
                "gap_loss": gap_loss.item(),
                "context_loss": context_loss.item(),
                "grad_loss": 0.0,
            }

        elif loss == "masked_l1_grad":
            total_loss, gap_loss, context_loss, grad_loss = masked_l1_grad_loss(
                pred, target, mask, context_weight=context_weight
            )
            loss_dict = {
                "loss": total_loss.item(),
                "gap_loss": gap_loss.item(),
                "context_loss": context_loss.item(),
                "grad_loss": grad_loss.item(),
            }

        elif loss == "masked_huber":
            total_loss, gap_loss, context_loss = masked_huber_loss(
                pred, target, mask, context_weight=context_weight
            )
            loss_dict = {
                "loss": total_loss.item(),
                "gap_loss": gap_loss.item(),
                "context_loss": context_loss.item(),
                "grad_loss": 0.0,
            }

        elif loss == "masked_huber_grad":
            total_loss, gap_loss, context_loss, grad_loss = masked_huber_grad_loss(
                pred, target, mask, context_weight=context_weight
            )
            loss_dict = {
                "loss": total_loss.item(),
                "gap_loss": gap_loss.item(),
                "context_loss": context_loss.item(),
                "grad_loss": grad_loss.item(),
            }

        elif loss == "masked_multires_l1":
            total_loss, gap_loss, context_loss = masked_multires_l1_loss(
                pred, target, mask, context_weight=context_weight
            )
            loss_dict = {
                "loss": total_loss.item(),
                "gap_loss": gap_loss.item(),
                "context_loss": context_loss.item(),
                "grad_loss": 0.0,
            }

        elif loss == "masked_multires_l1_grad":
            total_loss, gap_loss, context_loss, grad_loss = masked_multires_l1_grad_loss(
                pred, target, mask, context_weight=context_weight
            )
            loss_dict = {
                "loss": total_loss.item(),
                "gap_loss": gap_loss.item(),
                "context_loss": context_loss.item(),
                "grad_loss": grad_loss.item(),
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
        scheduler_min_lr=1e-6,
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
    def train_epoch(self, dataloader, device, grad_clip=1.0, context_weight=0.1):
        self.train()
        running = self.make_running_dict()
        num_batches = 0

        for batch in tqdm(dataloader, leave=False, desc="UNet Train"):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            self.optimiser.zero_grad(set_to_none=True)

            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                pred = self(x, mask)
                total_loss, loss_dict = self.compute_loss(
                    pred, y, mask,
                    loss=self.loss_name,
                    context_weight=context_weight
                )

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
    def evaluate(self, dataloader, device, context_weight=0.1):
        self.eval()
        running = self.make_running_dict()
        num_batches = 0

        for batch in tqdm(dataloader, leave=False, desc="UNet Val"):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                pred = self(x, mask)
                total_loss, loss_dict = self.compute_loss(
                    pred, y, mask,
                    loss=self.loss_name,
                    context_weight=context_weight
                )

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
        # context weight decay (VIAI-inspired)
        context_weight_decay=True,
        fixed_context_weight=0.05,
        cw_initial=0.12,
        cw_decay_rate=0.9,
        cw_floor=0.03,
        cw_decay_every=8,
        # resume support for disconnects
        start_epoch=1,
        resume_checkpoint_path=None,
        load_history=True,
    ):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)

        self.history = {
            "epoch": [],
            "lr": [],
            "context_weight": [],
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

        # resume from checkpoint
        if resume_checkpoint_path is not None:
            resume_checkpoint_path = Path(resume_checkpoint_path)
            if not resume_checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {resume_checkpoint_path}")

            ckpt = torch.load(resume_checkpoint_path, map_location=device)

            if "model_state_dict" in ckpt:
                self.load_state_dict(ckpt["model_state_dict"])
            else:
                raise KeyError("Checkpoint does not contain 'model_state_dict'")

            if "optimiser_state_dict" in ckpt and self.optimiser is not None:
                self.optimiser.load_state_dict(ckpt["optimiser_state_dict"])

            ckpt_epoch = ckpt.get("epoch", 0)
            start_epoch = ckpt_epoch + 1

            if "best_score" in ckpt:
                manager.best_score = ckpt["best_score"]
            if "best_epoch" in ckpt:
                manager.best_epoch = ckpt["best_epoch"]

            if verbose:
                print(f"Resumed from checkpoint: {resume_checkpoint_path}")
                print(f"Checkpoint epoch: {ckpt_epoch}")
                print(f"Will continue from epoch: {start_epoch}")
                print(f"Recovered best score: {manager.best_score}")
                print(f"Recovered best epoch: {manager.best_epoch}")

        # load existing history csv if present
        if load_history and history_path.exists():
            old_history_df = pd.read_csv(history_path)
            missing_cols = [k for k in self.history.keys() if k not in old_history_df.columns]
            if len(missing_cols) == 0:
                self.history = {col: old_history_df[col].tolist() for col in self.history.keys()}
                if verbose:
                    print(f"Loaded existing history from: {history_path}")
                    print(f"Existing history length: {len(old_history_df)} epochs")
            else:
                if verbose:
                    print(f"History file exists but columns do not fully match. Missing: {missing_cols}")
                    print("Starting a fresh history dictionary instead.")

        # main epoch loop
        for epoch in tqdm(range(start_epoch, n_epochs + 1), desc="Training U-Net"):

            # compute context weight for this epoch
            if context_weight_decay:
                cw = get_context_weight(
                    epoch,
                    initial=cw_initial,
                    decay_rate=cw_decay_rate,
                    floor=cw_floor,
                    decay_every=cw_decay_every,
                )
            else:
                cw = fixed_context_weight

            # train / eval
            train_metrics = self.train_epoch(
                train_loader, device=device, grad_clip=grad_clip, context_weight=cw
            )
            val_metrics = self.evaluate(
                val_loader, device=device, context_weight=cw
            )

            current_lr = self.get_lr()

            epoch_record = {
                "epoch": epoch,
                "lr": current_lr,
                "context_weight": cw,
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

            # save csv every epoch (survives disconnects)
            pd.DataFrame(self.history).to_csv(history_path, index=False)

            # epoch summary
            if verbose:
                print(f"Epoch {epoch:02d}  (context_weight: {cw:.4f})")
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

            # best / last saving and early stopping
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
