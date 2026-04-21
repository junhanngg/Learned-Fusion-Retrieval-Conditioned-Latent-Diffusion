import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from utils.checkpoint import ModelCheckpoint
from utils.losses import (
    diffusion_noise_mse_loss,
    diffusion_noise_l1_loss,
    diffusion_noise_huber_loss,
    masked_diffusion_noise_mse_loss,
    per_sample_mse_loss,
    per_sample_masked_mse_loss,
    min_snr,
    masked_latent_l1_loss,
    masked_latent_l2_loss,
)

# diffusion schedule container
@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    posterior_variance: torch.Tensor


# main diffusion u-net
class DiffusionU_Net(nn.Module):
    def __init__(
        self,
        latent_channels=8,
        base_channels=128,
        time_dim=256,
        num_steps=1000,
        schedule_s=0.008,
        schedule_max_beta=0.999,
        schedule_device=None,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.time_dim = time_dim
        self.num_steps = num_steps
        self.schedule_s = schedule_s
        self.schedule_max_beta = schedule_max_beta

        # input/output channels
        self.in_channels = 3 * latent_channels + 1
        self.out_channels = latent_channels

        # compiled/training state
        self.optimiser = None
        self.compiled_config = {}
        self.schedule = None

        if schedule_device is not None:
            self.schedule = self.cosine_schedule(
                num_steps=num_steps,
                s=schedule_s,
                max_beta=schedule_max_beta,
                device=schedule_device,
            )

        # time embedding
        self.time_embed = self.SinusoidalTimeEmb(time_dim)
        self.time_mlp = self.TimeEmbMLP(time_dim)

        # input projection
        self.in_conv = nn.Conv2d(self.in_channels, base_channels, kernel_size=3, padding=1)

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

        # output
        self.out_norm = self.make_norm(base_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_channels, self.out_channels, kernel_size=3, padding=1)

    # =====================================================
    # nested building blocks
    # =====================================================
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

    class TimeEmbMLP(nn.Module):
        def __init__(self, time_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(time_dim, time_dim * 4),
                nn.SiLU(),
                nn.Linear(time_dim * 4, time_dim),
            )

        def forward(self, t_emb):
            return self.net(t_emb)

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
            return self.conv(x)

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

    class SelfAttention(nn.Module):
        def __init__(self, channels, num_heads=4):
            super().__init__()
            assert channels % num_heads == 0, "channels must be divisible by num_heads"
            self.channels = channels
            self.num_heads = num_heads
            self.head_dim = channels // num_heads

            self.norm = DiffusionU_Net.make_norm(channels)
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

    class DilatedResBlock(nn.Module):
        def __init__(self, channels, dilation=2):
            super().__init__()
            padding = dilation
            self.norm1 = DiffusionU_Net.make_norm(channels)
            self.act1 = nn.SiLU()
            self.conv1 = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
            )

            self.norm2 = DiffusionU_Net.make_norm(channels)
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

    class DiffusionResBlock(nn.Module):
        def __init__(self, in_channels, out_channels, time_emb_dim=None, use_se=True):
            super().__init__()
            self.norm1 = DiffusionU_Net.make_norm(in_channels)
            self.act1 = nn.SiLU()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

            self.time_proj = nn.Linear(time_emb_dim, out_channels * 2)

            self.norm2 = DiffusionU_Net.make_norm(out_channels)
            self.act2 = nn.SiLU()
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

            self.se = DiffusionU_Net.SEBlock(out_channels) if use_se else nn.Identity()
            self.skip = (
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
                if in_channels != out_channels
                else nn.Identity()
            )

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

    # =====================================================
    # basic tensor helpers
    # =====================================================
    @staticmethod
    def padding(x, multiple=8):
        _, _, h, w = x.shape
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
    def extract(a, t, x_shape):
        b = t.shape[0]
        out = a.gather(0, t)
        reshape_dims = (b,) + (1,) * (len(x_shape) - 1)
        return out.view(*reshape_dims)

    @staticmethod
    def encode2latentmean(vae, x):
        with torch.no_grad():
            mu, logvar = vae.encode(x)
        return mu

    @staticmethod
    def decode_from_latent(vae, z):
        with torch.no_grad():
            x_hat = vae.decode(z)
        return x_hat

    def match_spatial(self, x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="nearest")
        return x

    # =====================================================
    # schedule helpers
    # =====================================================
    @staticmethod
    def cosine_schedule(num_steps, s=0.008, max_beta=0.999, device="cpu"):
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
            dim=0,
        )
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        return DiffusionSchedule(
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
            sqrt_alpha_bars=sqrt_alpha_bars,
            sqrt_one_minus_alpha_bars=sqrt_one_minus_alpha_bars,
            sqrt_recip_alphas=sqrt_recip_alphas,
            posterior_variance=posterior_variance,
        )

    def build_schedule(self, device=None, num_steps=None):
        if device is None:
            device = next(self.parameters()).device
        if num_steps is None:
            num_steps = self.num_steps

        self.schedule = self.cosine_schedule(
            num_steps=num_steps,
            s=self.schedule_s,
            max_beta=self.schedule_max_beta,
            device=device,
        )
        return self.schedule

    # =====================================================
    # diffusion math helpers
    # =====================================================
    def q_sample(self, z_0, t, noise, schedule=None):
        schedule = self.schedule if schedule is None else schedule
        sqrt_alpha_bar_t = self.extract(schedule.sqrt_alpha_bars, t, z_0.shape)
        sqrt_one_minus_alpha_bar_t = self.extract(schedule.sqrt_one_minus_alpha_bars, t, z_0.shape)
        return sqrt_alpha_bar_t * z_0 + sqrt_one_minus_alpha_bar_t * noise

    def make_known_latent(self, z_known_0, t, schedule=None, noise=None):
        schedule = self.schedule if schedule is None else schedule
        if noise is None:
            noise = torch.randn_like(z_known_0)
        return self.q_sample(z_known_0, t, noise, schedule)

    @staticmethod
    def preserve_known_region(z_sample, z_known, mask_latent):
        return mask_latent * z_sample + (1.0 - mask_latent) * z_known

    def predict_x0(self, z_t, eps_hat, t, schedule=None):
        schedule = self.schedule if schedule is None else schedule
        sqrt_alpha_bar_t = self.extract(schedule.sqrt_alpha_bars, t, z_t.shape)
        sqrt_one_minus_alpha_bar_t = self.extract(schedule.sqrt_one_minus_alpha_bars, t, z_t.shape)
        return (z_t - sqrt_one_minus_alpha_bar_t * eps_hat) / (sqrt_alpha_bar_t + 1e-8)

    # =====================================================
    # keras-style compile
    def compile(
        self,
        optimiser,
        schedule=None,
        noise_loss="mse",
        latent_loss="l1",
        latent_loss_weight=0.0,
        delta=1.0,
        masked_weight=3.0,
        use_min_snr=True,
        min_snr_gamma=5.0,
        grad_clip=1.0,
        use_amp=True,
        self_condition_prob=0.5,
        cond_dropout_prob=0.1,
    ):
        self.optimiser = optimiser

        if schedule is not None:
            self.schedule = schedule
        elif self.schedule is None:
            self.build_schedule(device=next(self.parameters()).device)

        self.compiled_config = {
            "noise_loss": noise_loss,
            "latent_loss": latent_loss,
            "latent_loss_weight": latent_loss_weight,
            "delta": delta,
            "masked_weight": masked_weight,
            "use_min_snr": use_min_snr,
            "min_snr_gamma": min_snr_gamma,
            "grad_clip": grad_clip,
            "use_amp": use_amp,
            "self_condition_prob": self_condition_prob,
            "cond_dropout_prob": cond_dropout_prob,
        }
        return self

    # forward
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

    # losses
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
        schedule = self.schedule if schedule is None else schedule

        if lossfn == "mse":
            if use_min_snr:
                per_sample = per_sample_mse_loss(pred_noise, true_noise)
                weight = min_snr(schedule, t, gamma=min_snr_gamma)
                return (per_sample * weight).mean()
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

    # core train/eval step
    def diffusion_step(
        self,
        vae,
        batch,
        device,
        schedule=None,
        noise_loss=None,
        latent_loss=None,
        latent_loss_weight=None,
        delta=None,
        masked_weight=None,
        use_min_snr=None,
        min_snr_gamma=None,
        self_condition_prob=None,
        cond_dropout_prob=None,
    ):
        if self.schedule is None and schedule is None:
            self.build_schedule(device=device)

        schedule = self.schedule if schedule is None else schedule

        cfg = self.compiled_config
        noise_loss = cfg.get("noise_loss", "mse") if noise_loss is None else noise_loss
        latent_loss = cfg.get("latent_loss", "l1") if latent_loss is None else latent_loss
        latent_loss_weight = cfg.get("latent_loss_weight", 0.0) if latent_loss_weight is None else latent_loss_weight
        delta = cfg.get("delta", 1.0) if delta is None else delta
        masked_weight = cfg.get("masked_weight", 3.0) if masked_weight is None else masked_weight
        use_min_snr = cfg.get("use_min_snr", True) if use_min_snr is None else use_min_snr
        min_snr_gamma = cfg.get("min_snr_gamma", 5.0) if min_snr_gamma is None else min_snr_gamma
        self_condition_prob = cfg.get("self_condition_prob", 0.5) if self_condition_prob is None else self_condition_prob
        cond_dropout_prob = cfg.get("cond_dropout_prob", 0.1) if cond_dropout_prob is None else cond_dropout_prob

        x = batch["x"].to(device)
        y = batch["y"].to(device)
        m = batch["mask"].to(device)

        x, _ = self.padding(x, multiple=8)
        y, _ = self.padding(y, multiple=8)
        m, _ = self.padding(m, multiple=8)

        with torch.no_grad():
            z_clean = self.encode2latentmean(vae, y)
            z_masked = self.encode2latentmean(vae, x)

        m_latent = F.interpolate(m, size=z_clean.shape[-2:], mode="nearest")

        if cond_dropout_prob > 0.0:
            keep = (
                (torch.rand(z_clean.shape[0], device=device) > cond_dropout_prob)
                .float()
                .view(-1, 1, 1, 1)
            )
            z_masked = z_masked * keep
            m_latent = m_latent * keep

        t = torch.randint(
            low=0,
            high=schedule.betas.shape[0],
            size=(z_clean.shape[0],),
            device=device,
        ).long()

        noise_unknown = torch.randn_like(z_clean)
        noise_known = torch.randn_like(z_masked)

        z_clean_t = self.q_sample(z_clean, t, noise_unknown, schedule)
        z_known_t = self.q_sample(z_masked, t, noise_known, schedule)

        z_t = m_latent * z_clean_t + (1.0 - m_latent) * z_known_t
        noise_target = m_latent * noise_unknown + (1.0 - m_latent) * noise_known

        z0_selfcond = torch.zeros_like(z_clean)
        if self_condition_prob > 0.0 and torch.rand(1).item() < self_condition_prob:
            with torch.no_grad():
                first_input = torch.cat([z_t, z_masked, z0_selfcond, m_latent], dim=1)
                first_pred = self(first_input, t)
                z0_selfcond = self.predict_x0(z_t, first_pred, t, schedule).detach()

        model_input = torch.cat([z_t, z_masked, z0_selfcond, m_latent], dim=1)
        eps_hat = self(model_input, t)

        noise_loss_value = self.compute_noiseloss(
            pred_noise=eps_hat,
            true_noise=noise_target,
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

            if latent_loss == "l1":
                latent_loss_value, latent_gap_loss, latent_context_loss = masked_latent_l1_loss(
                    pred_latent=z0_hat,
                    target_latent=z_clean,
                    mask_latent=m_latent,
                    context_weight=0.1,
                )
            elif latent_loss == "l2":
                latent_loss_value, latent_gap_loss, latent_context_loss = masked_latent_l2_loss(
                    pred_latent=z0_hat,
                    target_latent=z_clean,
                    mask_latent=m_latent,
                    context_weight=0.1,
                )
            else:
                raise ValueError(f"Unsupported latent auxiliary loss: {latent_loss}")
        else:
            z0_hat = None
            latent_loss_value = torch.tensor(0.0, device=device)
            latent_gap_loss = torch.tensor(0.0, device=device)
            latent_context_loss = torch.tensor(0.0, device=device)

        total_loss = noise_loss_value + latent_loss_weight * latent_loss_value

        return {
            "loss": total_loss,
            "noise_loss": noise_loss_value.detach(),
            "latent_loss": latent_loss_value.detach(),
            "latent_gap_loss": latent_gap_loss.detach(),
            "latent_context_loss": latent_context_loss.detach(),
            "z_clean": z_clean.detach(),
            "z_masked": z_masked.detach(),
            "m_latent": m_latent.detach(),
            "z0_hat": None if z0_hat is None else z0_hat.detach(),
        }

    # p_sample and inference
    @torch.no_grad()
    def p_sample(
        self,
        z_t,
        z_masked,
        mask_latent,
        z0_selfcond,
        t,
        schedule=None,
        add_noise=True,
    ):
        schedule = self.schedule if schedule is None else schedule
        model_input = torch.cat([z_t, z_masked, z0_selfcond, mask_latent], dim=1)
        eps_hat = self(model_input, t)

        beta_t = self.extract(schedule.betas, t, z_t.shape)
        alpha_t = self.extract(schedule.alphas, t, z_t.shape)
        alpha_bar_t = self.extract(schedule.alpha_bars, t, z_t.shape)
        posterior_var_t = self.extract(schedule.posterior_variance, t, z_t.shape)

        t_prev = torch.clamp(t - 1, min=0)
        alpha_bar_prev_t = self.extract(schedule.alpha_bars, t_prev, z_t.shape)

        is_t0 = (t == 0).float().view(z_t.shape[0], 1, 1, 1)
        alpha_bar_prev_t = is_t0 * torch.ones_like(alpha_bar_prev_t) + (1.0 - is_t0) * alpha_bar_prev_t

        z0_hat = self.predict_x0(z_t, eps_hat, t, schedule)
        z0_hat = z0_hat.clamp(-4.0, 4.0)

        coef1 = beta_t * torch.sqrt(alpha_bar_prev_t) / (1.0 - alpha_bar_t + 1e-8)
        coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t + 1e-8)
        model_mean = coef1 * z0_hat + coef2 * z_t

        if add_noise:
            noise = torch.randn_like(z_t)
            nonzero_mask = (t != 0).float().view(z_t.shape[0], 1, 1, 1)
            z_prev = model_mean + nonzero_mask * torch.sqrt(torch.clamp(posterior_var_t, min=1e-20)) * noise
        else:
            z_prev = model_mean

        return z_prev, z0_hat, eps_hat

    @torch.no_grad()
    def infer_diffusion_latent(
        self,
        vae,
        schedule,
        x_masked,
        mask,
        device,
        num_steps=None,
        add_noise=True,
        return_all_steps=False,
        use_self_conditioning=False,
    ):
        vae.eval()
        self.eval()

        x_masked = x_masked.to(device)
        mask = mask.to(device)

        orig_h, orig_w = x_masked.shape[-2:]

        x_masked, _ = self.padding(x_masked, multiple=8)
        mask, _ = self.padding(mask, multiple=8)

        z_masked = self.encode2latentmean(vae, x_masked)
        mask_latent = F.interpolate(mask, size=z_masked.shape[-2:], mode="nearest")

        T = schedule.betas.shape[0]
        if num_steps is None:
            num_steps = T
        if num_steps > T:
            raise ValueError(f"num_steps={num_steps} exceeds schedule length T={T}")

        known_noise = torch.randn_like(z_masked)

        t_init = torch.full(
            (z_masked.shape[0],),
            num_steps - 1,
            device=device,
            dtype=torch.long,
        )
        z_known_init = self.q_sample(z_masked, t_init, known_noise, schedule)

        z_t = torch.randn_like(z_masked)
        z_t = self.preserve_known_region(z_t, z_known_init, mask_latent)

        z0_selfcond = torch.zeros_like(z_masked)

        all_latents = []
        all_x0 = []

        if return_all_steps:
            all_latents.append(z_t.detach().cpu())
            all_x0.append(z0_selfcond.detach().cpu())

        for step in tqdm(reversed(range(num_steps)), total=num_steps, desc="Sampling Diffusion", leave=False):
            t = torch.full((z_t.shape[0],), step, device=device, dtype=torch.long)
            sc_input = z0_selfcond if use_self_conditioning else torch.zeros_like(z0_selfcond)

            z_prev, z0_hat, eps_hat = self.p_sample(
                z_t=z_t,
                z_masked=z_masked,
                mask_latent=mask_latent,
                z0_selfcond=sc_input,
                t=t,
                schedule=schedule,
                add_noise=add_noise,
            )

            if use_self_conditioning:
                z0_selfcond = z0_hat.detach()

            if step > 0:
                t_prev = torch.full((z_t.shape[0],), step - 1, device=device, dtype=torch.long)
                z_known_prev = self.q_sample(z_masked, t_prev, known_noise, schedule)
                z_t = self.preserve_known_region(z_prev, z_known_prev, mask_latent)
            else:
                z_t = self.preserve_known_region(z0_hat, z_masked, mask_latent)

            if return_all_steps:
                all_latents.append(z_t.detach().cpu())
                all_x0.append(z0_hat.detach().cpu())

        z_final = z_t
        x_hat = self.decode_from_latent(vae, z_final)
        x_hat = x_hat[..., :orig_h, :orig_w]

        return {
            "z_masked": z_masked,
            "mask_latent": mask_latent,
            "z_final": z_final,
            "x_hat": x_hat,
            "all_latents": all_latents if return_all_steps else None,
            "all_x0": all_x0 if return_all_steps else None,
        }

    # training helpers
    @staticmethod
    def lr_warmup(epoch, target_lr, warmup_epochs=5, start_lr=1e-6):
        if warmup_epochs <= 0:
            return target_lr
        progress = min(epoch / warmup_epochs, 1.0)
        return start_lr + progress * (target_lr - start_lr)

    @staticmethod
    def get_lr(optimiser):
        return optimiser.param_groups[0]["lr"]

    def train_epoch(self, vae, dataloader, device):
        if self.optimiser is None:
            raise RuntimeError("Call model.compile(...) before train_epoch().")

        super().train(True)

        cfg = self.compiled_config
        use_amp = cfg.get("use_amp", True) and ("cuda" in str(device))
        grad_clip = cfg.get("grad_clip", 1.0)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        running = {"loss": 0.0, "noise_loss": 0.0, "latent_loss": 0.0}
        n_batches = 0

        for batch in tqdm(dataloader, desc="Train Diffusion", leave=False):
            self.optimiser.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = self.diffusion_step(
                    vae=vae,
                    batch=batch,
                    device=device,
                    schedule=self.schedule,
                )

            scaler.scale(out["loss"]).backward()

            if grad_clip is not None:
                scaler.unscale_(self.optimiser)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip)

            scaler.step(self.optimiser)
            scaler.update()

            running["loss"] += out["loss"].item()
            running["noise_loss"] += out["noise_loss"].item()
            running["latent_loss"] += out["latent_loss"].item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    @torch.no_grad()
    def evaluate_epoch(self, vae, dataloader, device):
        super().eval()

        cfg = self.compiled_config
        use_amp = cfg.get("use_amp", True) and ("cuda" in str(device))

        running = {"loss": 0.0, "noise_loss": 0.0, "latent_loss": 0.0}
        n_batches = 0

        for batch in tqdm(dataloader, desc="Eval Diffusion", leave=False):
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = self.diffusion_step(
                    vae=vae,
                    batch=batch,
                    device=device,
                    schedule=self.schedule,
                    self_condition_prob=0.0,
                    cond_dropout_prob=0.0,
                )

            running["loss"] += out["loss"].item()
            running["noise_loss"] += out["noise_loss"].item()
            running["latent_loss"] += out["latent_loss"].item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    def fit(
        self,
        vae,
        train_loader,
        val_loader,
        device,
        n_epochs,
        checkpoint_dir,
        history_dir,

        monitor="val_loss",
        mode="min",
        patience=8,
        min_delta=1e-4,
        save_best_after_epoch=1,

        use_scheduler=True,
        scheduler_type="plateau",
        scheduler_factor=0.5,
        scheduler_patience=3,
        scheduler_min_lr=1e-7,

        start_epoch=1,
        resume_checkpoint_path=None,
        load_history=False,
        resume_scheduler=False,

        use_lr_warmup=True,
        warmup_epochs=5,
        base_lr=1e-4,
        warmup_start_lr=1e-6,
    ):
        if self.optimiser is None:
            raise RuntimeError("Call model.compile(...) before fit().")

        checkpoint_dir = Path(checkpoint_dir)
        history_dir = Path(history_dir)

        if self.schedule is None:
            self.build_schedule(device=device)

        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False

        history = {
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
            verbose=True,
        )

        if resume_checkpoint_path is not None:
            resume_checkpoint_path = Path(resume_checkpoint_path)
            if not resume_checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {resume_checkpoint_path}")

            ckpt = torch.load(resume_checkpoint_path, map_location=device)

            if "model_state_dict" in ckpt:
                self.load_state_dict(ckpt["model_state_dict"])
            else:
                raise KeyError("Checkpoint does not contain 'model_state_dict'")

            if "optimiser_state_dict" in ckpt:
                self.optimiser.load_state_dict(ckpt["optimiser_state_dict"])

            ckpt_epoch = ckpt.get("epoch", 0)
            start_epoch = ckpt_epoch + 1

            if "best_score" in ckpt:
                manager.best_score = ckpt["best_score"]
            if "best_epoch" in ckpt:
                manager.best_epoch = ckpt["best_epoch"]

            print(f"Resumed from checkpoint: {resume_checkpoint_path}")
            print(f"Checkpoint epoch: {ckpt_epoch}")
            print(f"Will continue from epoch: {start_epoch}")
            print(f"Recovered best score: {manager.best_score}")
            print(f"Recovered best epoch: {manager.best_epoch}")

        if load_history and history_dir.exists():
            old_history_df = pd.read_csv(history_dir)
            missing_cols = [k for k in history.keys() if k not in old_history_df.columns]

            if len(missing_cols) == 0:
                history = {col: old_history_df[col].tolist() for col in old_history_df.columns}
                print(f"Loaded existing history from: {history_dir}")
                print(f"Existing history length: {len(old_history_df)} epochs")
            else:
                print("History file exists but columns do not fully match current format.")
                print("Starting fresh history instead.")

        scheduler = None
        if use_scheduler:
            if scheduler_type == "plateau":
                scheduler = ReduceLROnPlateau(
                    self.optimiser,
                    mode=mode,
                    factor=scheduler_factor,
                    patience=scheduler_patience,
                    min_lr=scheduler_min_lr,
                )
            elif scheduler_type == "cosine":
                remaining_epochs = max(n_epochs - start_epoch + 1, 1)
                scheduler_tmax = remaining_epochs if resume_scheduler else n_epochs
                scheduler = CosineAnnealingLR(
                    self.optimiser,
                    T_max=scheduler_tmax,
                    eta_min=scheduler_min_lr,
                )
            else:
                raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

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
                dataloader=train_loader,
                device=device,
            )

            val_metrics = self.evaluate_epoch(
                vae=vae,
                dataloader=val_loader,
                device=device,
            )

            if scheduler is not None:
                if scheduler_type == "plateau":
                    if monitor == "val_loss":
                        scheduler.step(val_metrics["loss"])
                    elif monitor == "val_noise_loss":
                        scheduler.step(val_metrics["noise_loss"])
                    elif monitor == "val_latent_loss":
                        scheduler.step(val_metrics["latent_loss"])
                    else:
                        scheduler.step(val_metrics["loss"])
                elif scheduler_type == "cosine":
                    scheduler.step()

            current_lr = self.get_lr(self.optimiser)

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

            for key in history:
                history[key].append(epoch_record[key])

            history_df = pd.DataFrame(history)
            history_df.to_csv(history_dir, index=False)

            print(f"Epoch {epoch:02d}")
            print(f"Learning Rate:     {current_lr:.8e}")
            print(f"Train Loss:        {train_metrics['loss']:.6f}")
            print(f"Val Loss:          {val_metrics['loss']:.6f}")
            print(f"Train Noise Loss:  {train_metrics['noise_loss']:.6f}")
            print(f"Val Noise Loss:    {val_metrics['noise_loss']:.6f}")
            print(f"Train Latent Loss: {train_metrics['latent_loss']:.6f}")
            print(f"Val Latent Loss:   {val_metrics['latent_loss']:.6f}")
            print("-" * 60)

            # left untouched as requested
            manager.step(
                epoch=epoch,
                metrics=epoch_record,
                model=self,
                optimiser=self.optimiser,
            )

            if manager.should_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        return {
            "history": history,
            "best_score": manager.best_score,
            "best_epoch": manager.best_epoch,
            "checkpoint_dir": checkpoint_dir,
        }