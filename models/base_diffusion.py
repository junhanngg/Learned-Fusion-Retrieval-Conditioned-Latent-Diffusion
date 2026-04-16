import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# project root setup
notebook_dir = Path.cwd()
project_root = notebook_dir.parent
sys.path.insert(0, str(project_root))

# local imports
from utils.checkpoint import ModelCheckpoint
from utils.losses import (
    diffusion_noise_mse_loss,
    masked_diffusion_noise_mse_loss,
    diffusion_noise_l1_loss,
    diffusion_noise_huber_loss,
    latent_l1_loss,
    latent_l2_loss,
    per_sample_mse_loss,
    per_sample_masked_mse_loss,
    min_snr,
)

"""
Usage: 
--------------------------------------------------------------------------
- Create Model:
diffusion_unet = DiffusionUNet(
    latent_channels=4,
    base_channels=128,
    time_dim=256,
).to(device)
--------------------------------------------------------------------------
- Build optimiser:
diffusion_optimiser = AdamW(
    diffusion_unet.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
)
--------------------------------------------------------------------------
- Compile model:
diffusion_unet.compile(
    optimiser=diffusion_optimiser,
    device=device,
    use_amp=True,
    use_scheduler=True,
    scheduler_type="plateau",
    scheduler_mode="min",
    scheduler_factor=0.5,
    scheduler_patience=3,
    scheduler_min_lr=1e-7,
)
--------------------------------------------------------------------------
- Train model:
schedule = DiffusionUNet.cosine_schedule(
    num_steps=1000,
    device=device,
)
diffusion_results = diffusion_unet.fit(
    vae=vae,
    schedule=schedule,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    n_epochs=50,
    checkpoint_dir=root_dir / "diffusion_" / "checkpoints",
    history_path=root_dir / "diffusion_" / "history" / "history.csv",

    noise_loss="masked_mse",
    latent_loss="l1",
    latent_loss_weight=0.05,
    delta=1.0,
    masked_weight=3.0,
    use_min_snr=True,
    min_snr_gamma=5.0,

    grad_clip=1.0,
    monitor="val_loss",
    mode="min",
    patience=10,
    min_delta=1e-4,
    save_best_after_epoch=1,

    verbose=True,
    resume_checkpoint_path=root_dir / "diffusion_" / "checkpoints" / "last_model.pt",
    load_history=True,
    resume_scheduler=False,

    use_lr_warmup=False,
    warmup_epochs=5,
    base_lr=1e-4,
    warmup_start_lr=1e-6,
)
"""

# Diffusion U-Net for latent-space spectrogram inpainting
class DiffusionUNet(nn.Module):
    # schedule container
    @dataclass
    class DiffusionSchedule:
        betas: torch.Tensor
        alphas: torch.Tensor
        alpha_bars: torch.Tensor
        sqrt_alpha_bars: torch.Tensor
        sqrt_one_minus_alpha_bars: torch.Tensor
        sqrt_recip_alphas: torch.Tensor
        posterior_variance: torch.Tensor

    # helper layers
    @staticmethod
    def make_norm(channels, max_groups=8):
        groups = min(max_groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)

    class SinusoidalTimeEmb(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, t):
            half_dim = self.dim // 2
            freq_factor = math.log(10000) / max(half_dim - 1, 1)
            freqs = torch.exp(
                torch.arange(half_dim, device=t.device, dtype=torch.float32) * (-freq_factor)
            )
            args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

            if self.dim % 2 == 1:
                emb = F.pad(emb, (0, 1))

            return emb

    class DownSample(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

        def forward(self, x):
            return self.conv(x)

    class UpSample(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        def forward(self, x):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = self.conv(x)
            return x

    class SelfAttention(nn.Module):
        def __init__(self, channels, num_heads=4):
            super().__init__()
            assert channels % num_heads == 0, "channels must be divisible by num_heads"

            self.channels = channels
            self.num_heads = num_heads
            self.head_dim = channels // num_heads

            self.norm = DiffusionUNet.make_norm(channels)

            self.to_q = nn.Conv2d(channels, channels, kernel_size=1)
            self.to_k = nn.Conv2d(channels, channels, kernel_size=1)
            self.to_v = nn.Conv2d(channels, channels, kernel_size=1)
            self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

        def forward(self, x):
            b, c, h, w = x.shape
            residual = x

            x = self.norm(x)

            q = self.to_q(x)
            k = self.to_k(x)
            v = self.to_v(x)

            q = q.view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
            k = k.view(b, self.num_heads, self.head_dim, h * w)
            v = v.view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)

            attn_scores = torch.matmul(q, k) / math.sqrt(self.head_dim)
            attn_weights = torch.softmax(attn_scores, dim=-1)

            out = torch.matmul(attn_weights, v)
            out = out.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
            out = self.proj_out(out)

            return out + residual

    class SEBlock(nn.Module):
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

    class DiffusionResBlock(nn.Module):
        def __init__(self, in_channels, out_channels, time_emb_dim=None, use_se=True):
            super().__init__()

            self.norm1 = DiffusionUNet.make_norm(in_channels)
            self.act1 = nn.SiLU()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

            self.time_proj = nn.Linear(time_emb_dim, out_channels * 2)

            self.norm2 = DiffusionUNet.make_norm(out_channels)
            self.act2 = nn.SiLU()
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

            self.se = DiffusionUNet.SEBlock(out_channels) if use_se else nn.Identity()
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        def forward(self, x, t_emb):
            residual = self.skip(x)

            h = self.norm1(x)
            h = self.act1(h)
            h = self.conv1(h)

            film = self.time_proj(t_emb)
            scale, shift = torch.chunk(film, 2, dim=1)

            scale = scale[:, :, None, None]
            shift = shift[:, :, None, None]

            h = self.norm2(h)
            h = h * (1.0 + scale) + shift
            h = self.act2(h)
            h = self.conv2(h)
            h = self.se(h)

            return h + residual

    class DilatedResBlock(nn.Module):
        def __init__(self, channels, dilation=2):
            super().__init__()
            padding = dilation

            self.norm1 = DiffusionUNet.make_norm(channels)
            self.act1 = nn.SiLU()
            self.conv1 = nn.Conv2d(
                channels, channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation
            )

            self.norm2 = DiffusionUNet.make_norm(channels)
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

    class TimeEmbMLP(nn.Module):
        """
        Small MLP to refine sinusoidal timestep embeddings.
        """
        def __init__(self, time_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(time_dim, time_dim * 4),
                nn.SiLU(),
                nn.Linear(time_dim * 4, time_dim),
            )

        def forward(self, t_emb):
            return self.net(t_emb)

    # model initialisation
    def __init__(self, latent_channels=4, base_channels=128, time_dim=256):
        super().__init__()

        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.time_dim = time_dim

        in_channels = 2 * latent_channels + 1
        out_channels = latent_channels

        self.time_embed = self.SinusoidalTimeEmb(time_dim)
        self.time_mlp = self.TimeEmbMLP(time_dim)

        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # encoder
        self.down1_block1 = self.DiffusionResBlock(base_channels, base_channels, time_emb_dim=time_dim, use_se=True)
        self.down1_block2 = self.DiffusionResBlock(base_channels, base_channels, time_emb_dim=time_dim, use_se=True)
        self.down1 = self.DownSample(base_channels)

        self.down2_block1 = self.DiffusionResBlock(base_channels, base_channels * 2, time_emb_dim=time_dim, use_se=True)
        self.down2_block2 = self.DiffusionResBlock(base_channels * 2, base_channels * 2, time_emb_dim=time_dim, use_se=True)
        self.down2 = self.DownSample(base_channels * 2)

        self.down3_block1 = self.DiffusionResBlock(base_channels * 2, base_channels * 4, time_emb_dim=time_dim, use_se=True)
        self.down3_block2 = self.DiffusionResBlock(base_channels * 4, base_channels * 4, time_emb_dim=time_dim, use_se=True)
        self.down3 = self.DownSample(base_channels * 4)

        # bottleneck
        self.mid_block1 = self.DiffusionResBlock(base_channels * 4, base_channels * 4, time_emb_dim=time_dim, use_se=True)
        self.mid_dilate1 = self.DilatedResBlock(base_channels * 4, dilation=2)
        self.mid_attn = self.SelfAttention(base_channels * 4, num_heads=4)
        self.mid_dilate2 = self.DilatedResBlock(base_channels * 4, dilation=4)
        self.mid_block2 = self.DiffusionResBlock(base_channels * 4, base_channels * 4, time_emb_dim=time_dim, use_se=True)

        # decoder
        self.up3 = self.UpSample(base_channels * 4)
        self.up3_block1 = self.DiffusionResBlock(base_channels * 8, base_channels * 4, time_emb_dim=time_dim, use_se=True)
        self.up3_block2 = self.DiffusionResBlock(base_channels * 4, base_channels * 2, time_emb_dim=time_dim, use_se=True)

        self.up2 = self.UpSample(base_channels * 2)
        self.up2_block1 = self.DiffusionResBlock(base_channels * 4, base_channels * 2, time_emb_dim=time_dim, use_se=True)
        self.up2_block2 = self.DiffusionResBlock(base_channels * 2, base_channels, time_emb_dim=time_dim, use_se=True)

        self.up1 = self.UpSample(base_channels)
        self.up1_block1 = self.DiffusionResBlock(base_channels * 2, base_channels, time_emb_dim=time_dim, use_se=True)
        self.up1_block2 = self.DiffusionResBlock(base_channels, base_channels, time_emb_dim=time_dim, use_se=True)

        self.out_norm = self.make_norm(base_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

        # training state
        self.optimiser = None
        self.scheduler = None
        self.device_obj = None
        self.device_type = "cpu"
        self.use_amp = False
        self.scaler = None
        self.scheduler_type = None

        self.history = None
        self.best_score = None
        self.best_epoch = None

    # general helpers
    @staticmethod
    def padding(x, multiple=16):
        """
        pad H and W so they are divisible by 16.
        """
        b, c, h, w = x.shape
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple

        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        pad_info = {
            "orig_h": h,
            "orig_w": w,
            "pad_h": pad_h,
            "pad_w": pad_w,
        }
        return x_pad, pad_info

    @staticmethod
    def unpadding(x, pad_info):
        return x[..., :pad_info["orig_h"], :pad_info["orig_w"]]

    @staticmethod
    def cosine_schedule(num_steps, s=0.008, max_beta=0.999, device="cpu"):
        """
        cosine beta schedule.
        """
        steps = num_steps + 1
        t = torch.linspace(0, num_steps, steps, device=device, dtype=torch.float32)
        t = t / num_steps

        f_t = torch.cos(((t + s) / (1 + s)) * math.pi * 0.5) ** 2
        alpha_bars = f_t / f_t[0]

        betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.clamp(betas, min=1e-8, max=max_beta)

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        sqrt_alpha_bars = torch.sqrt(alpha_bars)
        sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        alpha_bars_prev = torch.cat(
            [torch.tensor([1.0], device=device), alpha_bars[:-1]],
            dim=0
        )
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        return DiffusionUNet.DiffusionSchedule(
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
            sqrt_alpha_bars=sqrt_alpha_bars,
            sqrt_one_minus_alpha_bars=sqrt_one_minus_alpha_bars,
            sqrt_recip_alphas=sqrt_recip_alphas,
            posterior_variance=posterior_variance,
        )

    @staticmethod
    def extract(a, t, x_shape):
        b = t.shape[0]
        out = a.gather(0, t)
        reshape_dims = (b,) + (1,) * (len(x_shape) - 1)
        return out.view(*reshape_dims)

    @staticmethod
    def q_sample(z_0, t, noise, schedule):
        sqrt_alpha_bar_t = DiffusionUNet.extract(schedule.sqrt_alpha_bars, t, z_0.shape)
        sqrt_one_minus_alpha_bar_t = DiffusionUNet.extract(schedule.sqrt_one_minus_alpha_bars, t, z_0.shape)
        z_t = sqrt_alpha_bar_t * z_0 + sqrt_one_minus_alpha_bar_t * noise
        return z_t

    @staticmethod
    def predict_x0(z_t, eps_hat, t, schedule):
        sqrt_alpha_bar_t = DiffusionUNet.extract(schedule.sqrt_alpha_bars, t, z_t.shape)
        sqrt_one_minus_alpha_bar_t = DiffusionUNet.extract(schedule.sqrt_one_minus_alpha_bars, t, z_t.shape)
        z0_hat = (z_t - sqrt_one_minus_alpha_bar_t * eps_hat) / (sqrt_alpha_bar_t + 1e-8)
        return z0_hat

    @staticmethod
    def lr_warmup(epoch, target_lr, warmup_epochs=5, start_lr=1e-6):
        if warmup_epochs <= 0:
            return target_lr
        progress = min(epoch / warmup_epochs, 1.0)
        lr = start_lr + progress * (target_lr - start_lr)
        return lr

    def match_spatial(self, x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="nearest")
        return x

    # model forward
    def forward(self, x, t):
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)

        x0 = self.in_conv(x)

        d1 = self.down1_block1(x0, t_emb)
        d1 = self.down1_block2(d1, t_emb)
        x1 = self.down1(d1)

        d2 = self.down2_block1(x1, t_emb)
        d2 = self.down2_block2(d2, t_emb)
        x2 = self.down2(d2)

        d3 = self.down3_block1(x2, t_emb)
        d3 = self.down3_block2(d3, t_emb)
        x3 = self.down3(d3)

        h = self.mid_block1(x3, t_emb)
        h = self.mid_dilate1(h)
        h = self.mid_attn(h)
        h = self.mid_dilate2(h)
        h = self.mid_block2(h, t_emb)

        h = self.up3(h)
        h = self.match_spatial(h, d3)
        h = torch.cat([h, d3], dim=1)
        h = self.up3_block1(h, t_emb)
        h = self.up3_block2(h, t_emb)

        h = self.up2(h)
        h = self.match_spatial(h, d2)
        h = torch.cat([h, d2], dim=1)
        h = self.up2_block1(h, t_emb)
        h = self.up2_block2(h, t_emb)

        h = self.up1(h)
        h = self.match_spatial(h, d1)
        h = torch.cat([h, d1], dim=1)
        h = self.up1_block1(h, t_emb)
        h = self.up1_block2(h, t_emb)

        h = self.out_norm(h)
        h = self.out_act(h)
        out = self.out_conv(h)

        return out

    # loss configurations
    def compute_noiseloss(
        self,
        pred_noise,
        true_noise,
        lossfn="mse",
        delta=1.0,
        mask_latent=None,
        masked_weight=3.0,
        schedule=None,
        t=None,
        use_min_snr=False,
        min_snr_gamma=5.0,
    ):
        if lossfn == "mse":
            if use_min_snr:
                per_sample = per_sample_mse_loss(pred_noise, true_noise)
                weight = min_snr(schedule, t, gamma=min_snr_gamma)
                return (per_sample * weight).mean()
            else:
                return diffusion_noise_mse_loss(pred_noise, true_noise)

        elif lossfn == "masked_mse":
            if mask_latent is None:
                raise ValueError("mask_latent must be provided when using lossfn='masked_mse'")

            if use_min_snr:
                per_sample = per_sample_masked_mse_loss(
                    pred=pred_noise,
                    target=true_noise,
                    mask_latent=mask_latent,
                    masked_weight=masked_weight,
                )
                weight = min_snr(schedule, t, gamma=min_snr_gamma)
                return (per_sample * weight).mean()
            else:
                return masked_diffusion_noise_mse_loss(
                    pred_noise=pred_noise,
                    true_noise=true_noise,
                    mask_latent=mask_latent,
                    masked_weight=masked_weight,
                )

        elif lossfn == "l1":
            return diffusion_noise_l1_loss(pred_noise, true_noise)

        elif lossfn == "huber":
            return diffusion_noise_huber_loss(pred_noise, true_noise, delta=delta)

        else:
            raise ValueError(f"Unsupported diffusion noise loss: {lossfn}")

    @staticmethod
    def compute_latentloss(pred_latent, target_latent, lossfn="l1"):
        if lossfn == "l1":
            return latent_l1_loss(pred_latent, target_latent)
        elif lossfn == "l2":
            return latent_l2_loss(pred_latent, target_latent)
        else:
            raise ValueError(f"Unsupported latent auxiliary loss: {lossfn}")

    # one diffusion training step
    def diffusion_step(
        self,
        vae,
        schedule,
        batch,
        device,
        noise_loss="mse",
        latent_loss="l1",
        latent_loss_weight=0.0,
        delta=1.0,
        masked_weight=3.0,
        use_min_snr=True,
        min_snr_gamma=5.0,
    ):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        m = batch["mask"].to(device)

        x, _ = self.padding(x, multiple=16)
        y, _ = self.padding(y, multiple=16)
        m, _ = self.padding(m, multiple=16)

        with torch.no_grad():
            # use posterior mean mu as latent target/condition
            z_clean = vae.encode(y)[0]
            z_masked = vae.encode(x)[0]

        m_latent = F.interpolate(m, size=z_clean.shape[-2:], mode="nearest")

        noise = torch.randn_like(z_clean)

        t = torch.randint(
            low=0,
            high=schedule.betas.shape[0],
            size=(z_clean.shape[0],),
            device=device
        ).long()

        z_t = self.q_sample(z_clean, t, noise, schedule)

        model_input = torch.cat([z_t, z_masked, m_latent], dim=1)
        eps_hat = self(model_input, t)

        noise_loss_value = self.compute_noiseloss(
            pred_noise=eps_hat,
            true_noise=noise,
            lossfn=noise_loss,
            delta=delta,
            mask_latent=m_latent,
            masked_weight=masked_weight,
            schedule=schedule,
            t=t,
            use_min_snr=use_min_snr,
            min_snr_gamma=min_snr_gamma,
        )

        if latent_loss_weight > 0.0:
            z0_hat = self.predict_x0(z_t, eps_hat, t, schedule)
            latent_loss_value = self.compute_latentloss(
                pred_latent=z0_hat,
                target_latent=z_clean,
                lossfn=latent_loss,
            )
        else:
            latent_loss_value = torch.tensor(0.0, device=device)

        total_loss = noise_loss_value + latent_loss_weight * latent_loss_value

        return {
            "loss": total_loss,
            "noise_loss": noise_loss_value.detach(),
            "latent_loss": latent_loss_value.detach(),
        }

    # compile
    def compile(
        self,
        optimiser,
        device,
        use_amp=True,
        use_scheduler=True,
        scheduler_type="plateau",
        scheduler_mode="min",
        scheduler_factor=0.5,
        scheduler_patience=3,
        scheduler_min_lr=1e-7,
        cosine_tmax=50,
    ):
        self.optimiser = optimiser

        self.device_obj = device if isinstance(device, torch.device) else torch.device(device)
        self.device_type = self.device_obj.type
        self.to(self.device_obj)

        self.use_amp = use_amp and (self.device_type == "cuda")
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_amp)

        self.scheduler = None
        if use_scheduler:
            if scheduler_type == "plateau":
                self.scheduler = ReduceLROnPlateau(
                    self.optimiser,
                    mode=scheduler_mode,
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
        self.scheduler_mode = scheduler_mode

    # one epoch train loop
    def train_epoch(
        self,
        vae,
        schedule,
        dataloader,
        device,
        noise_loss="mse",
        latent_loss="l1",
        latent_loss_weight=0.0,
        delta=1.0,
        masked_weight=3.0,
        use_min_snr=True,
        min_snr_gamma=5.0,
        grad_clip=1.0,
    ):
        self.train()

        running = {
            "loss": 0.0,
            "noise_loss": 0.0,
            "latent_loss": 0.0,
        }
        n_batches = 0

        for batch in tqdm(dataloader, desc="Train Diffusion", leave=False):
            self.optimiser.zero_grad(set_to_none=True)

            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                out = self.diffusion_step(
                    vae=vae,
                    schedule=schedule,
                    batch=batch,
                    device=device,
                    noise_loss=noise_loss,
                    latent_loss=latent_loss,
                    latent_loss_weight=latent_loss_weight,
                    delta=delta,
                    masked_weight=masked_weight,
                    use_min_snr=use_min_snr,
                    min_snr_gamma=min_snr_gamma,
                )

            self.scaler.scale(out["loss"]).backward()

            if grad_clip is not None:
                self.scaler.unscale_(self.optimiser)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip)

            self.scaler.step(self.optimiser)
            self.scaler.update()

            running["loss"] += out["loss"].item()
            running["noise_loss"] += out["noise_loss"].item()
            running["latent_loss"] += out["latent_loss"].item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    # one epoch eval loop
    @torch.no_grad()
    def evaluate_epoch(
        self,
        vae,
        schedule,
        dataloader,
        device,
        noise_loss="mse",
        latent_loss="l1",
        latent_loss_weight=0.0,
        delta=1.0,
        masked_weight=3.0,
        use_min_snr=True,
        min_snr_gamma=5.0,
    ):
        self.eval()

        running = {
            "loss": 0.0,
            "noise_loss": 0.0,
            "latent_loss": 0.0,
        }
        n_batches = 0

        for batch in tqdm(dataloader, desc="Eval Diffusion", leave=False):
            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                out = self.diffusion_step(
                    vae=vae,
                    schedule=schedule,
                    batch=batch,
                    device=device,
                    noise_loss=noise_loss,
                    latent_loss=latent_loss,
                    latent_loss_weight=latent_loss_weight,
                    delta=delta,
                    masked_weight=masked_weight,
                    use_min_snr=use_min_snr,
                    min_snr_gamma=min_snr_gamma,
                )

            running["loss"] += out["loss"].item()
            running["noise_loss"] += out["noise_loss"].item()
            running["latent_loss"] += out["latent_loss"].item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    # fit
    def fit(
        self,
        vae,
        schedule,
        train_loader,
        val_loader,
        device,
        n_epochs,
        checkpoint_dir,
        history_path,
        noise_loss="masked_mse",
        latent_loss="l1",
        latent_loss_weight=0.05,
        delta=1.0,
        masked_weight=3.0,
        use_min_snr=True,
        min_snr_gamma=5.0,
        grad_clip=1.0,
        monitor="val_loss",
        mode="min",
        patience=8,
        min_delta=1e-4,
        save_best_after_epoch=1,
        verbose=True,
        resume_checkpoint_path=None,
        load_history=False,
        resume_scheduler=False,
        use_lr_warmup=True,
        warmup_epochs=5,
        base_lr=1e-4,
        warmup_start_lr=1e-6,
    ):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)

        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False

        self.history = {
            "epoch": [],
            "lr": [],
            "train_loss": [],
            "train_noise_loss": [],
            "train_latent_loss": [],
            "val_loss": [],
            "val_noise_loss": [],
            "val_latent_loss": [],
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

        start_epoch = 1

        if resume_checkpoint_path is not None:
            resume_checkpoint_path = Path(resume_checkpoint_path)
            if not resume_checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {resume_checkpoint_path}")

            ckpt = torch.load(resume_checkpoint_path, map_location=device)
            self.load_state_dict(ckpt["model_state_dict"])

            if self.optimiser is not None and "optimiser_state_dict" in ckpt:
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

        if load_history and history_path.exists():
            old_history_df = pd.read_csv(history_path)
            missing_cols = [k for k in self.history.keys() if k not in old_history_df.columns]

            if len(missing_cols) == 0:
                self.history = {col: old_history_df[col].tolist() for col in old_history_df.columns}
                if verbose:
                    print(f"Loaded existing history from: {history_path}")
                    print(f"Existing history length: {len(old_history_df)} epochs")

        if self.scheduler is not None and self.scheduler_type == "cosine" and resume_scheduler:
            remaining_epochs = max(n_epochs - start_epoch + 1, 1)
            self.scheduler = CosineAnnealingLR(
                self.optimiser,
                T_max=remaining_epochs,
                eta_min=self.scheduler.eta_min if hasattr(self.scheduler, "eta_min") else 1e-7,
            )

        for epoch in tqdm(range(start_epoch, n_epochs + 1), desc="Training Diffusion"):
            if use_lr_warmup and epoch <= warmup_epochs:
                warmup_lr = self.lr_warmup(
                    epoch=epoch,
                    target_lr=base_lr,
                    warmup_epochs=warmup_epochs,
                    start_lr=warmup_start_lr,
                )
                for param_group in self.optimiser.param_groups:
                    param_group["lr"] = warmup_lr

            train_metrics = self.train_epoch(
                vae=vae,
                schedule=schedule,
                dataloader=train_loader,
                device=device,
                noise_loss=noise_loss,
                latent_loss=latent_loss,
                latent_loss_weight=latent_loss_weight,
                delta=delta,
                masked_weight=masked_weight,
                use_min_snr=use_min_snr,
                min_snr_gamma=min_snr_gamma,
                grad_clip=grad_clip,
            )

            val_metrics = self.evaluate_epoch(
                vae=vae,
                schedule=schedule,
                dataloader=val_loader,
                device=device,
                noise_loss=noise_loss,
                latent_loss=latent_loss,
                latent_loss_weight=latent_loss_weight,
                delta=delta,
                masked_weight=masked_weight,
                use_min_snr=use_min_snr,
                min_snr_gamma=min_snr_gamma,
            )

            if self.scheduler is not None:
                if self.scheduler_type == "plateau":
                    if monitor == "val_loss":
                        self.scheduler.step(val_metrics["loss"])
                    elif monitor == "val_noise_loss":
                        self.scheduler.step(val_metrics["noise_loss"])
                    elif monitor == "val_latent_loss":
                        self.scheduler.step(val_metrics["latent_loss"])
                    else:
                        self.scheduler.step(val_metrics["loss"])
                elif self.scheduler_type == "cosine":
                    self.scheduler.step()

            current_lr = self.optimiser.param_groups[0]["lr"]

            epoch_record = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_noise_loss": train_metrics["noise_loss"],
                "train_latent_loss": train_metrics["latent_loss"],
                "val_loss": val_metrics["loss"],
                "val_noise_loss": val_metrics["noise_loss"],
                "val_latent_loss": val_metrics["latent_loss"],
            }

            for key in self.history:
                self.history[key].append(epoch_record[key])

            pd.DataFrame(self.history).to_csv(history_path, index=False)

            if verbose:
                print(f"Epoch {epoch:02d}")
                print(f"Learning Rate:     {current_lr:.8e}")
                print(f"Train Loss:        {train_metrics['loss']:.6f}")
                print(f"Val Loss:          {val_metrics['loss']:.6f}")
                print(f"Train Noise Loss:  {train_metrics['noise_loss']:.6f}")
                print(f"Val Noise Loss:    {val_metrics['noise_loss']:.6f}")
                print(f"Train Latent Loss: {train_metrics['latent_loss']:.6f}")
                print(f"Val Latent Loss:   {val_metrics['latent_loss']:.6f}")
                print("-" * 60)

            manager.step(
                epoch=epoch,
                metrics=epoch_record,
                model=self,
                optimiser=self.optimiser,
            )

            self.best_score = manager.best_score
            self.best_epoch = manager.best_epoch

            if manager.should_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        return {
            "history": self.history,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "checkpoint_dir": checkpoint_dir,
        }