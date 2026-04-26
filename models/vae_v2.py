# standard library
import os
import sys
import math
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
    psnr,
)

"""
Usage:
--------------------------------------------------------------------------
- Create Model:
vae = VAE(in_channels=1, base_channels=64, latent_channels=8).to(device)
--------------------------------------------------------------------------
- Get reconstruction loss function:
recon_lossfn = vae.get_loss(
    variant="long_gap",
    shortgap_loss="masked_l1_grad",
    longgap_loss="masked_multires_l1_grad",
)
print(recon_lossfn)
--------------------------------------------------------------------------
- Build optimiser:
optimiser = torch.optim.AdamW(vae.parameters(), lr=1e-4, weight_decay=1e-4)
--------------------------------------------------------------------------
- Compile model:
vae.compile(
    optimiser=optimiser,
    recon_lossfn=recon_lossfn,
    device=device,
    use_amp=True,
    use_scheduler=True,
    scheduler_type="plateau",
    scheduler_factor=0.3,
    scheduler_patience=4,
    scheduler_min_lr=1e-7,
)
--------------------------------------------------------------------------
- Train model:
results = vae.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    n_epochs=150,
    checkpoint_dir=vae_checkpoint_dir,
    history_path=vae_history_dir / "history.csv",
    beta_target=3e-4,
    use_kl_warmup=True,
    kl_warmup_epochs=10,
    masked_input_start=0.05,
    masked_input_end=0.3,
    masked_input_ramp_end=25,
    val_masked_input_prob=0.2,
    monitor="val_loss",
    mode="min",
    patience=8,
    save_best_after_epoch=1,
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


# padding helpers
def padding(x, multiple=8):
    """
    pad the last two dims (H, W) so they are divisible by `multiple`.
    VAE downsamples by powers of 2 (3 stages -> factor of 8), so both
    H and W must be divisible by 8 to avoid size mismatch issues.
    Padding is added to right and bottom, removed via `unpadding`.
    """
    _, _, h, w = x.shape

    # amount of padding needed for height/width
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple

    # pad format for 4d tensor
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    pad_info = {
        "orig_h": h,
        "orig_w": w,
        "pad_h": pad_h,
        "pad_w": pad_w,
    }
    return x_pad, pad_info


def unpadding(x, pad_info):
    """remove padding that was added by `padding`."""
    return x[..., :pad_info["orig_h"], :pad_info["orig_w"]]


# KL warmup schedule
def linear_kl_warmup(epoch, target_beta=5e-4, warmup_epochs=15, start_beta=0.0):
    """
    linear KL warmup: ramps beta_kl from start_beta to target_beta
    over `warmup_epochs` epochs, then holds at target_beta.
    """
    if warmup_epochs <= 0:
        return target_beta

    progress = min(epoch / warmup_epochs, 1.0)
    beta = start_beta + progress * (target_beta - start_beta)
    return beta


# masked input probability schedule
def masked_input_scheduler(epoch, start_prob=0.1, end_prob=0.8, ramp_end_epoch=50):
    """
    linearly increase masked-input probability from start_prob to end_prob
    until ramp_end_epoch, then keep it fixed.
    """
    if epoch >= ramp_end_epoch:
        return end_prob

    progress = (epoch - 1) / max(ramp_end_epoch - 1, 1)
    prob = start_prob + progress * (end_prob - start_prob)
    return prob


# choose what goes into VAE encoder
def choose_input(batch, device, masked_input_prob=0.7):
    """
    sometimes feed clean y, sometimes feed masked x;
    but always reconstruct clean y.
    """
    x = batch["x"].to(device, non_blocking=True)   # masked spectrogram
    y = batch["y"].to(device, non_blocking=True)   # clean spectrogram
    m = batch["mask"].to(device, non_blocking=True)

    use_masked = torch.rand(1).item() < masked_input_prob
    encoder_input = x if use_masked else y

    return encoder_input, y, m


# residual block: GN -> SiLU -> Conv -> GN -> SiLU -> Conv + skip
class ResidualBlock(nn.Module):
    """
    standard residual block: lets the model keep useful info and only
    change what is needed (similar to ResNet).
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        # first normalization + convolution path
        self.norm1 = make_norm(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = make_norm(out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.skip(x)

        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


# dilated residual block (used in bottleneck)
class DilatedResBlock(nn.Module):
    """
    dilated residual block: increases the receptive field in the bottleneck
    without further spatial compression.
    """
    def __init__(self, channels, dilation=2):
        super().__init__()

        self.norm1 = make_norm(channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )

        self.norm2 = make_norm(channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x

        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return h + residual


# downsampling: stride-2 conv -> halves spatial resolution
class DownSample(nn.Module):
    """halves H and W via stride-2 convolution."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# upsampling: nearest interp + conv -> doubles spatial resolution
class UpSample(nn.Module):
    """
    doubles H and W via nearest-neighbour interpolation + conv.
    using nearest+conv (rather than transposed conv) reduces checkerboard artefacts.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


# self-attention over spatial positions (used in bottleneck)
class SelfAttention(nn.Module):
    """
    multi-head self-attention over the spatial grid. typical U-Net design
    places self-attention at the bottleneck where the spatial dim is smallest.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = make_norm(channels)

        # q, k, v projections
        self.to_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_k = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x

        x = self.norm(x)

        q = self.to_q(x).view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        k = self.to_k(x).view(b, self.num_heads, self.head_dim, h * w)
        v = self.to_v(x).view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)

        attn_scores = torch.matmul(q, k) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
        out = self.proj_out(out)

        return out + residual


# encoder: spectrogram -> Gaussian latent (mu, logvar)
class Encoder(nn.Module):
    """
    deep encoder with 6 stages of processing and 3 downsampling operations.
    final compression is a factor of 8 spatially.
    """
    def __init__(self, in_channels=1, base_channels=64, latent_channels=8):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 4

        # initial projection from 1 channel to base feature channels
        self.in_conv = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

        # stage 1 - full resolution
        self.block1 = nn.Sequential(
            ResidualBlock(c1, c1),
            ResidualBlock(c1, c1),
        )
        self.down1 = DownSample(c1)  # 1st downsampling

        # stage 2 - half resolution
        self.block2 = nn.Sequential(
            ResidualBlock(c1, c2),
            ResidualBlock(c2, c2),
        )

        # stage 3 - more refinement before next compression
        self.block3 = nn.Sequential(
            ResidualBlock(c2, c2),
            ResidualBlock(c2, c2),
        )
        self.down2 = DownSample(c2)  # 2nd downsampling

        # stage 4 - quarter resolution
        self.stage4 = nn.Sequential(
            ResidualBlock(c2, c3),
            ResidualBlock(c3, c3),
        )

        # stage 5 - still quarter resolution
        self.stage5 = nn.Sequential(
            ResidualBlock(c3, c3),
            ResidualBlock(c3, c3),
        )
        self.down3 = DownSample(c3)  # 3rd downsampling

        # stage 6 - bottleneck entry
        self.stage6 = nn.Sequential(
            ResidualBlock(c3, c4),
            ResidualBlock(c4, c4),
        )

        # bottleneck
        self.mid = nn.Sequential(
            ResidualBlock(c4, c4),
            DilatedResBlock(c4, dilation=2),
            SelfAttention(c4, num_heads=4),
            DilatedResBlock(c4, dilation=4),
            ResidualBlock(c4, c4),
        )

        # gaussian posterior heads
        self.mu_head = nn.Conv2d(c4, latent_channels, kernel_size=1)
        self.logvar_head = nn.Conv2d(c4, latent_channels, kernel_size=1)

        # start latent posterior near a calm regime
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(self, x):
        x = self.in_conv(x)

        # stage 1
        x = self.block1(x)
        x = self.down1(x)

        # stages 2 and 3 at half resolution
        x = self.block2(x)
        x = self.block3(x)
        x = self.down2(x)

        # stages 4 and 5 at quarter resolution
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.down3(x)

        # stage 6 at eighth resolution
        x = self.stage6(x)

        # bottleneck
        x = self.mid(x)

        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        return mu, logvar


# decoder: latent -> reconstructed spectrogram (mirrors encoder)
class Decoder(nn.Module):
    """
    decoder mirrors the encoder architecture in reverse, taking a latent
    code and progressively upsampling to the original spectrogram size.
    """
    def __init__(self, out_channels=1, base_channels=64, latent_channels=8):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 4

        self.in_conv = nn.Conv2d(latent_channels, c4, kernel_size=3, padding=1)

        self.mid = nn.Sequential(
            ResidualBlock(c4, c4),
            SelfAttention(c4, num_heads=4),
            DilatedResBlock(c4, dilation=2),
            ResidualBlock(c4, c4),
        )

        # mirror of stage 6
        self.stage6 = nn.Sequential(
            ResidualBlock(c4, c4),
            ResidualBlock(c4, c4),
        )
        self.up3 = UpSample(c4)

        # mirror of stages 5 and 4
        self.stage5 = nn.Sequential(
            ResidualBlock(c4, c3),
            ResidualBlock(c3, c3),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(c3, c3),
            ResidualBlock(c3, c3),
        )
        self.up2 = UpSample(c3)

        # mirror of stages 3 and 2
        self.stage3 = nn.Sequential(
            ResidualBlock(c3, c2),
            ResidualBlock(c2, c2),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(c2, c2),
            ResidualBlock(c2, c2),
        )
        self.up1 = UpSample(c2)

        # mirror of stage 1
        self.stage1 = nn.Sequential(
            ResidualBlock(c2, c1),
            ResidualBlock(c1, c1),
        )

        self.out_norm = make_norm(c1)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.in_conv(z)
        x = self.mid(x)

        x = self.stage6(x)
        x = self.up3(x)

        x = self.stage5(x)
        x = self.stage4(x)
        x = self.up2(x)

        x = self.stage3(x)
        x = self.stage2(x)
        x = self.up1(x)

        x = self.stage1(x)

        x = self.out_norm(x)
        x = self.out_act(x)
        x = self.out_conv(x)
        return x


# main VAE model class
class VAE(nn.Module):
    """
    Variational Autoencoder for spectrogram inpainting.

    Architectural features:
    - 3 spatial downsamples in encoder (factor-8 compression in H and W)
    - Dilated residual blocks + self-attention in the bottleneck
    - Symmetric decoder with nearest-interp + conv upsamples
    - Gaussian posterior parameterised by 1x1 conv heads (mu, logvar)
    - Mirrored decoder reconstructs to original H, W via padding/unpadding helpers

    Architecture (factor-8 spatial compression):

    Input (B, 1, H, W)
      |
    Encoder
      | in_conv -> block1 -> down1 (H/2)
      | block2 -> block3 -> down2 (H/4)
      | stage4 -> stage5 -> down3 (H/8)
      | stage6
      | mid (Residual + Dilated + SelfAttention + Dilated + Residual)
      v
    (mu, logvar)  in latent_channels @ (H/8, W/8)
      |
    z = mu + sigma * eps    (reparameterisation)
      |
    Decoder
      | in_conv -> mid
      | stage6 -> up3 (H/4)
      | stage5 -> stage4 -> up2 (H/2)
      | stage3 -> stage2 -> up1 (H)
      | stage1 -> out_norm -> SiLU -> out_conv
      v
    Reconstructed spectrogram (B, 1, H, W)
    """
    def __init__(self, in_channels=1, base_channels=64, latent_channels=8):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
        )

        self.decoder = Decoder(
            out_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
        )

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_channels = latent_channels

        # training state
        self.history = None
        self.best_score = None
        self.best_epoch = None
        self.recon_lossfn = None
        self.optimiser = None
        self.scheduler = None
        self.scaler = None
        self.device_obj = None
        self.device_type = "cpu"
        self.use_amp = False
        self.scheduler_type = None

    # encode: x -> (mu, logvar)
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    # reparameterisation trick: z = mu + sigma * eps
    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    # decode: z -> reconstructed spectrogram
    def decode(self, z):
        return self.decoder(z)

    # full forward pass
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    # loss selection
    def get_loss(
        self,
        variant="long_gap",
        shortgap_loss="masked_l1_grad",
        longgap_loss="masked_multires_l1_grad",
    ):
        if variant == "short_gap":
            return shortgap_loss
        elif variant == "long_gap":
            return longgap_loss
        else:
            raise ValueError(f"Invalid gap variant {variant}.")

    # reconstruction-only component of the VAE loss
    def compute_reconloss(
        self,
        recon,
        target,
        mask,
        lossfn="masked_multires_l1_grad",
        context_weight=0.1,
        grad_weight=0.1,
        delta=1.0,
        scales=(1, 2, 4),
        scale_weights=None,
        eps=1e-8,
    ):
        if lossfn == "masked_l1":
            total_loss, gap_loss, context_loss = masked_l1_loss(
                pred=recon, target=target, mask=mask,
                context_weight=context_weight, eps=eps,
            )
            grad_loss = torch.tensor(0.0, device=recon.device)

        elif lossfn == "masked_huber":
            total_loss, gap_loss, context_loss = masked_huber_loss(
                pred=recon, target=target, mask=mask,
                context_weight=context_weight, delta=delta, eps=eps,
            )
            grad_loss = torch.tensor(0.0, device=recon.device)

        elif lossfn == "masked_l1_grad":
            total_loss, gap_loss, context_loss, grad_loss = masked_l1_grad_loss(
                pred=recon, target=target, mask=mask,
                context_weight=context_weight, grad_weight=grad_weight, eps=eps,
            )

        elif lossfn == "masked_huber_grad":
            total_loss, gap_loss, context_loss, grad_loss = masked_huber_grad_loss(
                pred=recon, target=target, mask=mask,
                context_weight=context_weight, grad_weight=grad_weight,
                delta=delta, eps=eps,
            )

        elif lossfn == "masked_multires_l1":
            total_loss, gap_loss, context_loss = masked_multires_l1_loss(
                pred=recon, target=target, mask=mask,
                context_weight=context_weight, scales=scales,
                scale_weights=scale_weights, eps=eps,
            )
            grad_loss = torch.tensor(0.0, device=recon.device)

        elif lossfn == "masked_multires_l1_grad":
            total_loss, gap_loss, context_loss, grad_loss = masked_multires_l1_grad_loss(
                pred=recon, target=target, mask=mask,
                context_weight=context_weight, grad_weight=grad_weight,
                scales=scales, scale_weights=scale_weights, eps=eps,
            )

        else:
            raise ValueError(f"Unsupported VAE reconstruction loss: {lossfn}")

        return {
            "recon_loss": total_loss,
            "gap_loss": gap_loss.detach(),
            "context_loss": context_loss.detach(),
            "grad_loss": grad_loss.detach(),
        }

    # full VAE loss = reconstruction + beta * KL
    def compute_loss(
        self,
        recon,
        target,
        mu,
        logvar,
        mask,
        beta_kl=5e-4,
        recon_lossfn="masked_multires_l1_grad",
        context_weight=0.1,
        grad_weight=0.1,
        delta=1.0,
        scales=(1, 2, 4),
        scale_weights=None,
        eps=1e-8,
    ):
        recon_dict = self.compute_reconloss(
            recon=recon,
            target=target,
            mask=mask,
            lossfn=recon_lossfn,
            context_weight=context_weight,
            grad_weight=grad_weight,
            delta=delta,
            scales=scales,
            scale_weights=scale_weights,
            eps=eps,
        )

        # force KL branch to float32 for numerical stability under AMP
        mu32 = mu.float()
        logvar32 = logvar.float()

        # clamp logvar before exp to avoid overflow
        logvar32 = torch.clamp(logvar32, min=-10.0, max=10.0)

        # KL divergence averaged over all latent elements
        kl_per_element = -0.5 * (1 + logvar32 - mu32.pow(2) - logvar32.exp())
        kl_loss = kl_per_element.mean()

        total_loss = recon_dict["recon_loss"] + beta_kl * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_dict["recon_loss"].detach(),
            "gap_loss": recon_dict["gap_loss"],
            "context_loss": recon_dict["context_loss"],
            "grad_loss": recon_dict["grad_loss"],
            "kl_loss": kl_loss.detach(),
        }

    # metrics
    def compute_metrics(self, recon, target, mask):
        return {
            "gap_mae": masked_mae(recon, target, mask).item(),
            "gap_rmse": masked_rmse(recon, target, mask).item(),
            "full_mae": full_mae(recon, target).item(),
            "full_rmse": full_rmse(recon, target).item(),
            "psnr": psnr(recon, target).item(),
        }

    def make_running_dict(self):
        return {
            "loss": 0.0,
            "recon_loss": 0.0,
            "gap_loss": 0.0,
            "context_loss": 0.0,
            "grad_loss": 0.0,
            "kl_loss": 0.0,
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
        recon_lossfn,
        device,
        use_amp=True,
        use_scheduler=True,
        scheduler_type="plateau",
        scheduler_factor=0.3,
        scheduler_patience=4,
        scheduler_min_lr=1e-7,
        cosine_tmax=150,
    ):
        self.optimiser = optimiser
        self.recon_lossfn = recon_lossfn

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
    def train_epoch(
        self,
        dataloader,
        device,
        beta_kl=5e-4,
        masked_input_prob=0.7,
        context_weight=0.1,
        grad_weight=0.1,
        delta=1.0,
        scales=(1, 2, 4),
        scale_weights=None,
        grad_clip=1.0,
    ):
        self.train()
        running = self.make_running_dict()
        n_batches = 0

        for batch in tqdm(dataloader, desc="VAE Train", leave=False):
            x_in, y, m = choose_input(
                batch=batch, device=device, masked_input_prob=masked_input_prob
            )

            # pad consistently for factor-8 model
            x_in, _ = padding(x_in, multiple=8)
            y, pad_info = padding(y, multiple=8)
            m, _ = padding(m, multiple=8)

            self.optimiser.zero_grad(set_to_none=True)

            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                recon, mu, logvar, z = self(x_in)

                loss_dict = self.compute_loss(
                    recon=recon,
                    target=y,
                    mu=mu,
                    logvar=logvar,
                    mask=m,
                    beta_kl=beta_kl,
                    recon_lossfn=self.recon_lossfn,
                    context_weight=context_weight,
                    grad_weight=grad_weight,
                    delta=delta,
                    scales=scales,
                    scale_weights=scale_weights,
                )

                # check for gradient explosion early
                if not torch.isfinite(loss_dict["loss"]):
                    print("Non-finite loss detected")
                    print("mu min/max/mean:", mu.min().item(), mu.max().item(), mu.mean().item())
                    print("logvar min/max/mean:", logvar.min().item(), logvar.max().item(), logvar.mean().item())
                    raise ValueError("Loss became non-finite")

            self.scaler.scale(loss_dict["loss"]).backward()

            if grad_clip is not None:
                self.scaler.unscale_(self.optimiser)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip)

            self.scaler.step(self.optimiser)
            self.scaler.update()

            # remove padding before metrics so values reflect original size
            with torch.no_grad():
                recon_eval = unpadding(recon.detach().float(), pad_info)
                y_eval = unpadding(y.float(), pad_info)
                m_eval = unpadding(m.float(), pad_info)
                metric_dict = self.compute_metrics(recon_eval, y_eval, m_eval)

            running["loss"] += loss_dict["loss"].item()
            running["recon_loss"] += loss_dict["recon_loss"].item()
            running["gap_loss"] += loss_dict["gap_loss"].item()
            running["context_loss"] += loss_dict["context_loss"].item()
            running["grad_loss"] += loss_dict["grad_loss"].item()
            running["kl_loss"] += loss_dict["kl_loss"].item()

            for k, v in metric_dict.items():
                running[k] += v

            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    # validation epoch with masked-input mixing (matches downstream diffusion use)
    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        device,
        beta_kl=5e-4,
        masked_input_prob=1.0,
        context_weight=0.1,
        grad_weight=0.1,
        delta=1.0,
        scales=(1, 2, 4),
        scale_weights=None,
    ):
        """
        masked_input_prob defaults to 1.0 because that is closer to how the
        encoder will actually be used inside the latent diffusion pipeline.
        """
        self.eval()
        running = self.make_running_dict()
        n_batches = 0

        for batch in tqdm(dataloader, desc="VAE Val", leave=False):
            x_in, y, m = choose_input(
                batch=batch, device=device, masked_input_prob=masked_input_prob,
            )

            x_in, _ = padding(x_in, multiple=8)
            y, pad_info = padding(y, multiple=8)
            m, _ = padding(m, multiple=8)

            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                recon, mu, logvar, z = self(x_in)

                loss_dict = self.compute_loss(
                    recon=recon,
                    target=y,
                    mu=mu,
                    logvar=logvar,
                    mask=m,
                    beta_kl=beta_kl,
                    recon_lossfn=self.recon_lossfn,
                    context_weight=context_weight,
                    grad_weight=grad_weight,
                    delta=delta,
                    scales=scales,
                    scale_weights=scale_weights,
                )

            recon_eval = unpadding(recon.detach().float(), pad_info)
            y_eval = unpadding(y.float(), pad_info)
            m_eval = unpadding(m.float(), pad_info)

            metric_dict = self.compute_metrics(recon_eval, y_eval, m_eval)

            running["loss"] += loss_dict["loss"].item()
            running["recon_loss"] += loss_dict["recon_loss"].item()
            running["gap_loss"] += loss_dict["gap_loss"].item()
            running["context_loss"] += loss_dict["context_loss"].item()
            running["grad_loss"] += loss_dict["grad_loss"].item()
            running["kl_loss"] += loss_dict["kl_loss"].item()

            for k, v in metric_dict.items():
                running[k] += v

            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    # validation epoch using only clean spectrograms (tokenizer-style)
    @torch.no_grad()
    def evaluate_clean(
        self,
        dataloader,
        device,
        beta_kl=5e-4,
        context_weight=0.1,
        grad_weight=0.1,
        delta=1.0,
        scales=(1, 2, 4),
        scale_weights=None,
    ):
        """
        Clean validation: encoder input = clean y, target = clean y.
        Tells you how good the VAE is as a tokenizer for clean spectrograms,
        independent of any masked-input behaviour.
        """
        self.eval()
        running = self.make_running_dict()
        n_batches = 0

        for batch in tqdm(dataloader, desc="VAE Val (clean)", leave=False):
            y = batch["y"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)

            y_in, _ = padding(y, multiple=8)
            y_target, pad_info = padding(y, multiple=8)
            m, _ = padding(m, multiple=8)

            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                recon, mu, logvar, z = self(y_in)

                loss_dict = self.compute_loss(
                    recon=recon,
                    target=y_target,
                    mu=mu,
                    logvar=logvar,
                    mask=m,
                    beta_kl=beta_kl,
                    recon_lossfn=self.recon_lossfn,
                    context_weight=context_weight,
                    grad_weight=grad_weight,
                    delta=delta,
                    scales=scales,
                    scale_weights=scale_weights,
                )

            recon_eval = unpadding(recon.detach().float(), pad_info)
            y_eval = unpadding(y_target.float(), pad_info)
            m_eval = unpadding(m.float(), pad_info)

            metric_dict = self.compute_metrics(recon_eval, y_eval, m_eval)

            running["loss"] += loss_dict["loss"].item()
            running["recon_loss"] += loss_dict["recon_loss"].item()
            running["gap_loss"] += loss_dict["gap_loss"].item()
            running["context_loss"] += loss_dict["context_loss"].item()
            running["grad_loss"] += loss_dict["grad_loss"].item()
            running["kl_loss"] += loss_dict["kl_loss"].item()

            for k, v in metric_dict.items():
                running[k] += v

            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    # fit
    def fit(
        self,
        train_loader,
        val_loader,
        device,
        n_epochs,
        checkpoint_dir,
        history_path,
        # KL warmup
        beta_target=5e-4,
        beta_start=0.0,
        use_kl_warmup=True,
        kl_warmup_epochs=15,
        # masked input schedule
        masked_input_start=0.1,
        masked_input_end=0.8,
        masked_input_ramp_end=50,
        val_masked_input_prob=0.5,
        # reconstruction loss settings
        context_weight=0.1,
        grad_weight=0.1,
        delta=1.0,
        scales=(1, 2, 4),
        scale_weights=None,
        # checkpointing / early stopping
        monitor="val_gap_rmse",
        mode="min",
        patience=8,
        min_delta=1e-4,
        save_best_after_epoch=3,
        grad_clip=1.0,
        verbose=True,
        # resume
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
            "masked_input_prob": [],
            "beta_kl": [],
            "lr": [],

            "train_loss": [],
            "train_recon_loss": [],
            "train_kl_loss": [],
            "train_gap_loss": [],
            "train_context_loss": [],
            "train_grad_loss": [],
            "train_gap_mae": [],
            "train_gap_rmse": [],
            "train_full_mae": [],
            "train_full_rmse": [],
            "train_psnr": [],

            "val_loss": [],
            "val_recon_loss": [],
            "val_kl_loss": [],
            "val_gap_loss": [],
            "val_context_loss": [],
            "val_grad_loss": [],
            "val_gap_mae": [],
            "val_gap_rmse": [],
            "val_full_mae": [],
            "val_full_rmse": [],
            "val_psnr": [],

            "clean_val_loss": [],
            "clean_val_recon_loss": [],
            "clean_val_kl_loss": [],
            "clean_val_gap_loss": [],
            "clean_val_context_loss": [],
            "clean_val_grad_loss": [],
            "clean_val_gap_mae": [],
            "clean_val_gap_rmse": [],
            "clean_val_full_mae": [],
            "clean_val_full_rmse": [],
            "clean_val_psnr": [],
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
            if resume_checkpoint_path.exists():
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
        for epoch in tqdm(range(start_epoch, n_epochs + 1), desc="Training VAE"):

            # KL beta for this epoch
            if use_kl_warmup:
                beta_this_epoch = linear_kl_warmup(
                    epoch=epoch,
                    target_beta=beta_target,
                    warmup_epochs=kl_warmup_epochs,
                    start_beta=beta_start,
                )
            else:
                beta_this_epoch = beta_target

            # masked-input probability for this epoch
            masked_input_prob = masked_input_scheduler(
                epoch=epoch,
                start_prob=masked_input_start,
                end_prob=masked_input_end,
                ramp_end_epoch=masked_input_ramp_end,
            )

            # train / val / clean val
            train_metrics = self.train_epoch(
                dataloader=train_loader,
                device=device,
                beta_kl=beta_this_epoch,
                masked_input_prob=masked_input_prob,
                context_weight=context_weight,
                grad_weight=grad_weight,
                delta=delta,
                scales=scales,
                scale_weights=scale_weights,
                grad_clip=grad_clip,
            )

            val_metrics = self.evaluate(
                dataloader=val_loader,
                device=device,
                beta_kl=beta_this_epoch,
                masked_input_prob=val_masked_input_prob,
                context_weight=context_weight,
                grad_weight=grad_weight,
                delta=delta,
                scales=scales,
                scale_weights=scale_weights,
            )

            clean_val_metrics = self.evaluate_clean(
                dataloader=val_loader,
                device=device,
                beta_kl=beta_this_epoch,
                context_weight=context_weight,
                grad_weight=grad_weight,
                delta=delta,
                scales=scales,
                scale_weights=scale_weights,
            )

            current_lr = self.get_lr()

            epoch_record = {
                "epoch": epoch,
                "masked_input_prob": masked_input_prob,
                "beta_kl": beta_this_epoch,
                "lr": current_lr,

                "train_loss": train_metrics["loss"],
                "train_recon_loss": train_metrics["recon_loss"],
                "train_kl_loss": train_metrics["kl_loss"],
                "train_gap_loss": train_metrics["gap_loss"],
                "train_context_loss": train_metrics["context_loss"],
                "train_grad_loss": train_metrics["grad_loss"],
                "train_gap_mae": train_metrics["gap_mae"],
                "train_gap_rmse": train_metrics["gap_rmse"],
                "train_full_mae": train_metrics["full_mae"],
                "train_full_rmse": train_metrics["full_rmse"],
                "train_psnr": train_metrics["psnr"],

                "val_loss": val_metrics["loss"],
                "val_recon_loss": val_metrics["recon_loss"],
                "val_kl_loss": val_metrics["kl_loss"],
                "val_gap_loss": val_metrics["gap_loss"],
                "val_context_loss": val_metrics["context_loss"],
                "val_grad_loss": val_metrics["grad_loss"],
                "val_gap_mae": val_metrics["gap_mae"],
                "val_gap_rmse": val_metrics["gap_rmse"],
                "val_full_mae": val_metrics["full_mae"],
                "val_full_rmse": val_metrics["full_rmse"],
                "val_psnr": val_metrics["psnr"],

                "clean_val_loss": clean_val_metrics["loss"],
                "clean_val_recon_loss": clean_val_metrics["recon_loss"],
                "clean_val_kl_loss": clean_val_metrics["kl_loss"],
                "clean_val_gap_loss": clean_val_metrics["gap_loss"],
                "clean_val_context_loss": clean_val_metrics["context_loss"],
                "clean_val_grad_loss": clean_val_metrics["grad_loss"],
                "clean_val_gap_mae": clean_val_metrics["gap_mae"],
                "clean_val_gap_rmse": clean_val_metrics["gap_rmse"],
                "clean_val_full_mae": clean_val_metrics["full_mae"],
                "clean_val_full_rmse": clean_val_metrics["full_rmse"],
                "clean_val_psnr": clean_val_metrics["psnr"],
            }

            # append to history
            for key in self.history:
                self.history[key].append(epoch_record[key])

            # save csv every epoch (survives disconnects)
            pd.DataFrame(self.history).to_csv(history_path, index=False)

            # epoch summary
            if verbose:
                print(f"Epoch {epoch:02d}")
                print(f"Masked Input Prob:    {masked_input_prob:.3f}")
                print(f"KL Beta:              {beta_this_epoch:.8f}")
                print(f"Learning Rate:        {current_lr:.8e}")
                print(f"Train Loss:           {train_metrics['loss']:.6f}")
                print(f"Train Gap RMSE:       {train_metrics['gap_rmse']:.6f}")
                print(f"Train PSNR:           {train_metrics['psnr']:.4f}")
                print(f"Masked Val Loss:      {val_metrics['loss']:.6f}")
                print(f"Masked Val Gap RMSE:  {val_metrics['gap_rmse']:.6f}")
                print(f"Masked Val PSNR:      {val_metrics['psnr']:.4f}")
                print(f"Clean Val Loss:       {clean_val_metrics['loss']:.6f}")
                print(f"Clean Val Gap RMSE:   {clean_val_metrics['gap_rmse']:.6f}")
                print(f"Clean Val PSNR:       {clean_val_metrics['psnr']:.4f}")
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
            "recon_lossfn": self.recon_lossfn,
        }

    # simple prediction helper: x -> reconstruction (with padding handled)
    @torch.no_grad()
    def predict_batch(self, x, device, use_mu=True):
        """
        encode-decode a batch of spectrograms.
        if use_mu=True, decodes from mu (deterministic);
        otherwise samples z via the reparameterisation trick.
        """
        self.eval()
        x = x.to(device)
        x_pad, pad_info = padding(x, multiple=8)
        mu, logvar = self.encode(x_pad)
        z = mu if use_mu else self.reparameterise(mu, logvar)
        recon = self.decode(z)
        return unpadding(recon, pad_info)