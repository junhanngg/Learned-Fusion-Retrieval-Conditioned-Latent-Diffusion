import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# project root setup
import sys
notebook_dir = Path.cwd()
project_root = notebook_dir.parent
sys.path.insert(0, str(project_root))

# local imports
from utils.checkpoint import ModelCheckpoint
from utils.losses import (
    # reconstruction losses
    masked_l1_loss,
    masked_huber_loss,
    masked_l1_grad_loss,
    masked_huber_grad_loss,
    masked_multires_l1_loss,
    masked_multires_l1_grad_loss,

    # evaluation metrics
    masked_mae,
    masked_rmse,
    full_mae,
    full_rmse,
    psnr
)

# VAE utilities
def make_norm(channels, max_groups=8):
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


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


def unpadding(x, pad_info):
    return x[..., :pad_info["orig_h"], :pad_info["orig_w"]]


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

# architecture blocks
class ResidualBlock(nn.Module):
    """
    GN -> SiLU -> Conv -> GN -> SiLU -> Conv + skip
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

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


class DilatedResBlock(nn.Module):
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

        self.norm = make_norm(channels)

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


# encoder / decoder
class Encoder(nn.Module):
    """
    deep encoder with 6 stages and 3 downsampling operations.
    final compression factor = 8.
    """
    def __init__(self, in_channels=1, base_channels=64, latent_channels=8):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 4

        self.in_conv = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

        self.block1 = nn.Sequential(
            ResidualBlock(c1, c1),
            ResidualBlock(c1, c1),
        )
        self.down1 = DownSample(c1)

        self.block2 = nn.Sequential(
            ResidualBlock(c1, c2),
            ResidualBlock(c2, c2),
        )

        self.block3 = nn.Sequential(
            ResidualBlock(c2, c2),
            ResidualBlock(c2, c2),
        )
        self.down2 = DownSample(c2)

        self.stage4 = nn.Sequential(
            ResidualBlock(c2, c3),
            ResidualBlock(c3, c3),
        )

        self.stage5 = nn.Sequential(
            ResidualBlock(c3, c3),
            ResidualBlock(c3, c3),
        )
        self.down3 = DownSample(c3)

        self.stage6 = nn.Sequential(
            ResidualBlock(c3, c4),
            ResidualBlock(c4, c4),
        )

        self.mid = nn.Sequential(
            ResidualBlock(c4, c4),
            DilatedResBlock(c4, dilation=2),
            SelfAttention(c4, num_heads=4),
            DilatedResBlock(c4, dilation=4),
            ResidualBlock(c4, c4),
        )

        self.mu_head = nn.Conv2d(c4, latent_channels, kernel_size=1)
        self.logvar_head = nn.Conv2d(c4, latent_channels, kernel_size=1)

        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(self, x):
        x = self.in_conv(x)

        x = self.block1(x)
        x = self.down1(x)

        x = self.block2(x)
        x = self.block3(x)
        x = self.down2(x)

        x = self.stage4(x)
        x = self.stage5(x)
        x = self.down3(x)

        x = self.stage6(x)
        x = self.mid(x)

        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        return mu, logvar


class Decoder(nn.Module):
    """
    Mirrors the encoder.
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

        self.stage6 = nn.Sequential(
            ResidualBlock(c4, c4),
            ResidualBlock(c4, c4),
        )
        self.up3 = UpSample(c4)

        self.stage5 = nn.Sequential(
            ResidualBlock(c4, c3),
            ResidualBlock(c3, c3),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(c3, c3),
            ResidualBlock(c3, c3),
        )
        self.up2 = UpSample(c3)

        self.stage3 = nn.Sequential(
            ResidualBlock(c3, c2),
            ResidualBlock(c2, c2),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(c2, c2),
            ResidualBlock(c2, c2),
        )
        self.up1 = UpSample(c2)

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
    
# GAN discriminator and losses
class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator for optional adversarial VAE training.
    """
    def __init__(self, in_channels=1, base_channels=64, n_layers=3):
        super().__init__()

        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        in_ch = base_channels

        for i in range(1, n_layers):
            out_ch = min(base_channels * (2 ** i), 512)
            layers.extend([
                nn.utils.spectral_norm(
                    nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
                ),
                make_norm(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_ch = out_ch

        out_ch = min(base_channels * (2 ** n_layers), 512)

        layers.extend([
            nn.utils.spectral_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=1, padding=1)
            ),
            make_norm(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        ])

        layers.append(
            nn.utils.spectral_norm(
                nn.Conv2d(out_ch, 1, kernel_size=4, stride=1, padding=1)
            )
        )

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


def discriminator_hinge_loss(real_scores, fake_scores):
    loss_real = F.relu(1.0 - real_scores).mean()
    loss_fake = F.relu(1.0 + fake_scores).mean()
    return 0.5 * (loss_real + loss_fake)


def generator_hinge_loss(fake_scores):
    return -fake_scores.mean()


def compute_adaptive_weight(recon_loss, gen_loss, last_layer_weight, eps=1e-4):
    recon_grads = torch.autograd.grad(
        recon_loss,
        last_layer_weight,
        retain_graph=True,
    )[0]

    gen_grads = torch.autograd.grad(
        gen_loss,
        last_layer_weight,
        retain_graph=True,
    )[0]

    weight = recon_grads.norm() / (gen_grads.norm() + eps)
    weight = torch.clamp(weight, 0.0, 1e4).detach()

    return weight


def multires_l1_loss(pred, target, scales=(1, 2, 4)):
    total = 0.0

    for s in scales:
        if s == 1:
            pred_s = pred
            target_s = target
        else:
            pred_s = F.avg_pool2d(pred, kernel_size=s, stride=s)
            target_s = F.avg_pool2d(target, kernel_size=s, stride=s)

        total = total + F.l1_loss(pred_s, target_s)

    return total / len(scales)


def vae_kl_loss(mu, logvar):
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


def vae_loss(
    recon,
    target,
    mu,
    logvar,
    beta_kl=1e-6,
    scales=(1, 2, 4),
    discriminator=None,
    last_layer_weight=None,
    adv_weight=0.3,
    use_adv=False,
):
    recon_loss = multires_l1_loss(recon, target, scales=scales)
    kl_loss = vae_kl_loss(mu, logvar)

    if use_adv and discriminator is not None:
        fake_scores = discriminator(recon)
        gen_loss = generator_hinge_loss(fake_scores)

        if last_layer_weight is not None:
            adaptive_weight = compute_adaptive_weight(
                recon_loss,
                gen_loss,
                last_layer_weight,
            )
        else:
            adaptive_weight = torch.tensor(1.0, device=recon.device)

        total_loss = recon_loss + beta_kl * kl_loss + adv_weight * adaptive_weight * gen_loss

    else:
        gen_loss = torch.tensor(0.0, device=recon.device)
        adaptive_weight = torch.tensor(0.0, device=recon.device)
        total_loss = recon_loss + beta_kl * kl_loss

    return {
        "loss": total_loss,
        "recon_loss": recon_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "gen_loss": gen_loss.detach(),
        "adaptive_weight": adaptive_weight.detach(),
    }


def linear_kl_warmup(epoch, target_beta=5e-5, warmup_epochs=10, start_beta=0.0):
    if warmup_epochs <= 0:
        return target_beta

    progress = min(epoch / warmup_epochs, 1.0)
    beta = start_beta + progress * (target_beta - start_beta)

    return beta


def get_lr(optimiser):
    return optimiser.param_groups[0]["lr"]


def compute_vae_metrics(recon_eval, y_eval):
    return {
        "full_mae": full_mae(recon_eval, y_eval).item(),
        "full_rmse": full_rmse(recon_eval, y_eval).item(),
        "psnr": psnr(recon_eval, y_eval).item(),
    }


def make_vae_running_dict():
    return {
        "loss": 0.0,
        "recon_loss": 0.0,
        "kl_loss": 0.0,

        "gen_loss": 0.0,
        "disc_loss": 0.0,
        "adaptive_weight": 0.0,

        "diagnostic_gen_adv": 0.0,
        "diagnostic_disc_real": 0.0,
        "diagnostic_disc_fake": 0.0,

        "full_mae": 0.0,
        "full_rmse": 0.0,
        "psnr": 0.0,
    }

# modular VAE class with compile / fit / evaluate
class VAE(nn.Module):
    """
    Representation VAE for log-mel spectrogram reconstruction.

    Main API:
        vae.compile(...)
        vae.fit(...)
        vae.evaluate(...)
        vae.reconstruct(...)
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

        self.latent_channels = latent_channels

        # filled by compile()
        self.optimiser = None
        self.discriminator = None
        self.disc_optimiser = None
        self.compile_config = {}
        self.is_compiled = False

    # core VAE methods
    def encode(self, x):
        return self.encoder(x)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, sample_posterior=True):
        mu, logvar = self.encode(x)

        if sample_posterior:
            z = self.reparameterise(mu, logvar)
        else:
            z = mu

        recon = self.decode(z)

        return recon, mu, logvar, z

    @torch.no_grad()
    def encode_latent_mean(self, x):
        mu, _ = self.encode(x)
        return mu

    @torch.no_grad()
    def reconstruct(self, x, use_mean=True, multiple=8):
        self.eval()

        x_pad, pad_info = padding(x, multiple=multiple)
        mu, logvar = self.encode(x_pad)

        if use_mean:
            z = mu
        else:
            z = self.reparameterise(mu, logvar)

        recon = self.decode(z)
        recon = unpadding(recon, pad_info)

        return recon

    # Keras-style compile
    def compile(
        self,
        optimiser=None,
        lr=1e-4,
        weight_decay=1e-4,

        discriminator=None,
        disc_optimiser=None,
        disc_lr=2.5e-5,
        disc_weight_decay=0.0,
        disc_betas=(0.5, 0.9),

        use_adv=False,
        adv_weight=0.3,
        use_amp=False,
    ):
        """
        Stores optimisers and adversarial config inside the VAE.

        You can either pass your own optimiser:

            vae.compile(optimiser=vae_optimiser)

        or let this create one:

            vae.compile(lr=1e-4, weight_decay=1e-4)
        """

        if optimiser is None:
            optimiser = AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

        self.optimiser = optimiser

        if discriminator is not None:
            self.discriminator = discriminator

        if use_adv and self.discriminator is not None and disc_optimiser is None:
            disc_optimiser = AdamW(
                self.discriminator.parameters(),
                lr=disc_lr,
                weight_decay=disc_weight_decay,
                betas=disc_betas,
            )

        self.disc_optimiser = disc_optimiser

        self.compile_config = {
            "use_adv": use_adv,
            "adv_weight": adv_weight,
            "use_amp": use_amp,
        }

        self.is_compiled = True

        return self

    # internal helpers
    def _resolve_compile_value(self, name, value):
        if value is not None:
            return value
        return self.compile_config.get(name)

    def _amp_settings(self, device, use_amp):
        use_amp = bool(use_amp) and ("cuda" in str(device))
        amp_device = "cuda" if "cuda" in str(device) else "cpu"
        return use_amp, amp_device

    def _make_history(self):
        return {
            "epoch": [],
            "beta_kl": [],
            "lr": [],
            "adv_active": [],

            "train_loss": [],
            "train_recon_loss": [],
            "train_kl_loss": [],
            "train_gen_loss": [],
            "train_disc_loss": [],
            "train_adaptive_weight": [],
            "train_full_mae": [],
            "train_full_rmse": [],
            "train_psnr": [],

            "val_loss": [],
            "val_recon_loss": [],
            "val_kl_loss": [],
            "val_full_mae": [],
            "val_full_rmse": [],
            "val_psnr": [],

            "val_diagnostic_gen_adv": [],
            "val_diagnostic_disc_real": [],
            "val_diagnostic_disc_fake": [],
        }

    def _load_history_if_possible(self, history, history_path, load_history):
        if not load_history:
            return history

        if not history_path.exists():
            return history

        old_history_df = pd.read_csv(history_path)

        missing_cols = [k for k in history.keys() if k not in old_history_df.columns]

        if len(missing_cols) == 0:
            history = {k: old_history_df[k].tolist() for k in history.keys()}
            print(f"Loaded existing history from: {history_path}")
        else:
            print(f"History schema changed, starting fresh history.")
            print(f"Missing cols: {missing_cols}")

        return history

    def _load_checkpoint_if_possible(
        self,
        resume_checkpoint_path,
        device,
        optimiser,
        discriminator,
        disc_optimiser,
        manager,
    ):
        start_epoch = 1

        if resume_checkpoint_path is None:
            return start_epoch

        resume_checkpoint_path = Path(resume_checkpoint_path)

        if not resume_checkpoint_path.exists():
            print(f"Resume checkpoint not found: {resume_checkpoint_path}")
            return start_epoch

        ckpt = torch.load(resume_checkpoint_path, map_location=device)

        self.load_state_dict(ckpt["model_state_dict"])

        if "optimiser_state_dict" in ckpt and ckpt["optimiser_state_dict"] is not None:
            optimiser.load_state_dict(ckpt["optimiser_state_dict"])

        if discriminator is not None:
            if "discriminator_state_dict" in ckpt and ckpt["discriminator_state_dict"] is not None:
                discriminator.load_state_dict(ckpt["discriminator_state_dict"])

            if (
                disc_optimiser is not None
                and "disc_optimiser_state_dict" in ckpt
                and ckpt["disc_optimiser_state_dict"] is not None
            ):
                disc_optimiser.load_state_dict(ckpt["disc_optimiser_state_dict"])

        start_epoch = ckpt.get("epoch", 0) + 1
        manager.best_score = ckpt.get("best_score", manager.best_score)
        manager.best_epoch = ckpt.get("best_epoch", manager.best_epoch)

        print(f"Resumed from checkpoint: {resume_checkpoint_path}")
        print(f"Continuing from epoch: {start_epoch}")

        return start_epoch

    def _make_scheduler(
        self,
        optimiser,
        use_scheduler,
        scheduler_type,
        mode,
        scheduler_factor,
        scheduler_patience,
        scheduler_min_lr,
        n_epochs,
    ):
        if not use_scheduler:
            return None

        if scheduler_type == "plateau":
            return ReduceLROnPlateau(
                optimiser,
                mode=mode,
                factor=scheduler_factor,
                patience=scheduler_patience,
                min_lr=scheduler_min_lr,
            )

        if scheduler_type == "cosine":
            return CosineAnnealingLR(
                optimiser,
                T_max=n_epochs,
                eta_min=scheduler_min_lr,
            )

        raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

    def _train_epoch(
        self,
        dataloader,
        optimiser,
        device,
        beta_kl,
        scales=(1, 2, 4),
        grad_clip=1.0,
        adv_weight=0.3,
        use_amp=True,
        discriminator=None,
        disc_optimiser=None,
        use_adv=False,
        adv_warmup_done=False,
    ):
        self.train()

        if discriminator is not None:
            discriminator.train()

        use_amp, amp_device = self._amp_settings(device, use_amp)

        scaler = torch.amp.GradScaler(device=amp_device, enabled=use_amp)
        disc_scaler = torch.amp.GradScaler(device=amp_device, enabled=use_amp)

        running = make_vae_running_dict()
        n_batches = 0

        use_adv_this_epoch = use_adv and adv_warmup_done

        for batch in tqdm(dataloader, desc="Training VAE"):
            y_clean = batch["y"].to(device, non_blocking=True)
            y_padded, pad_info = padding(y_clean, multiple=8)

            # step 1: train VAE / generator
            optimiser.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                recon_padded, mu, logvar, z = self(y_padded, sample_posterior=True)
                recon = unpadding(recon_padded, pad_info)

                last_layer_weight = (
                    self.decoder.out_conv.weight if use_adv_this_epoch else None
                )

                loss_dict = vae_loss(
                    recon=recon,
                    target=y_clean,
                    mu=mu,
                    logvar=logvar,
                    beta_kl=beta_kl,
                    scales=scales,
                    discriminator=discriminator,
                    adv_weight=adv_weight,
                    last_layer_weight=last_layer_weight,
                    use_adv=use_adv_this_epoch,
                )

                total_loss = loss_dict["loss"]

            scaler.scale(total_loss).backward()

            if grad_clip is not None:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

            scaler.step(optimiser)
            scaler.update()

            # step 2: train discriminator
            disc_loss_value = 0.0

            if use_adv_this_epoch and discriminator is not None and disc_optimiser is not None:
                disc_optimiser.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                    real_scores = discriminator(y_clean)
                    fake_scores = discriminator(recon.detach())
                    disc_loss = discriminator_hinge_loss(real_scores, fake_scores)

                disc_scaler.scale(disc_loss).backward()

                if grad_clip is not None:
                    disc_scaler.unscale_(disc_optimiser)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)

                disc_scaler.step(disc_optimiser)
                disc_scaler.update()

                disc_loss_value = disc_loss.item()

            # step 3: metrics
            with torch.no_grad():
                metric_dict = compute_vae_metrics(recon, y_clean)

            running["loss"] += total_loss.item()
            running["recon_loss"] += loss_dict["recon_loss"].item()
            running["kl_loss"] += loss_dict["kl_loss"].item()
            running["gen_loss"] += loss_dict["gen_loss"].item()
            running["disc_loss"] += disc_loss_value
            running["adaptive_weight"] += float(loss_dict["adaptive_weight"])

            for k, v in metric_dict.items():
                running[k] += v

            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        device,
        beta_kl=5e-5,
        adv_weight=None,
        scales=(1, 2, 4),
        use_amp=None,
        discriminator=None,
    ):
        self.eval()

        if discriminator is None:
            discriminator = self.discriminator

        if adv_weight is None:
            adv_weight = self.compile_config.get("adv_weight", 0.3)

        if use_amp is None:
            use_amp = self.compile_config.get("use_amp", False)

        if discriminator is not None:
            discriminator.eval()

        running = make_vae_running_dict()
        n_batches = 0

        use_amp, amp_device = self._amp_settings(device, use_amp)

        for batch in tqdm(dataloader, desc="Eval VAE", leave=False):
            y = batch["y"].to(device, non_blocking=True)
            y_padded, pad_info = padding(y, multiple=8)

            with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                recon, mu, logvar, z = self(y_padded, sample_posterior=False)

                loss_dict = vae_loss(
                    recon=recon,
                    target=y_padded,
                    mu=mu,
                    logvar=logvar,
                    adv_weight=adv_weight,
                    beta_kl=beta_kl,
                    scales=scales,
                    discriminator=None,
                    last_layer_weight=None,
                    use_adv=False,
                )

            recon_eval = unpadding(recon.detach(), pad_info)
            y_eval = unpadding(y_padded, pad_info)

            metric_dict = compute_vae_metrics(recon_eval, y_eval)

            if discriminator is not None:
                real_scores = discriminator(y_eval)
                fake_scores = discriminator(recon_eval)

                diagnostic_gen_adv = (-fake_scores.mean()).item()
                diagnostic_disc_real = real_scores.mean().item()
                diagnostic_disc_fake = fake_scores.mean().item()
            else:
                diagnostic_gen_adv = 0.0
                diagnostic_disc_real = 0.0
                diagnostic_disc_fake = 0.0

            running["loss"] += loss_dict["loss"].item()
            running["recon_loss"] += loss_dict["recon_loss"].item()
            running["kl_loss"] += loss_dict["kl_loss"].item()

            running["gen_loss"] += 0.0
            running["disc_loss"] += 0.0
            running["adaptive_weight"] += 0.0

            running["diagnostic_gen_adv"] += diagnostic_gen_adv
            running["diagnostic_disc_real"] += diagnostic_disc_real
            running["diagnostic_disc_fake"] += diagnostic_disc_fake

            for k, v in metric_dict.items():
                running[k] += v

            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    # full fit loop
    def fit(
        self,
        train_loader,
        val_loader,
        device,
        n_epochs,
        checkpoint_dir,
        history_path,

        # optional override
        optimiser=None,

        # KL warmup
        beta_target=5e-5,
        beta_start=0.0,
        use_kl_warmup=True,
        kl_warmup_epochs=10,

        # reconstruction loss
        scales=(1, 2, 4),

        # checkpointing + early stopping
        monitor="val_loss",
        mode="min",
        patience=10,
        min_delta=1e-4,
        save_best_after_epoch=1,

        # optimisation
        grad_clip=1.0,
        use_scheduler=True,
        scheduler_type="plateau",
        scheduler_factor=0.5,
        scheduler_patience=3,
        scheduler_min_lr=1e-7,

        # resume
        resume_checkpoint_path=None,
        load_history=True,

        # compile overrides
        use_amp=None,
        use_adv=None,
        discriminator=None,
        disc_optimiser=None,
        adv_warmup_epochs=20,
        adv_weight=None,
    ):
        if optimiser is None:
            optimiser = self.optimiser

        if optimiser is None:
            raise ValueError(
                "No optimiser found. Call vae.compile(optimiser=...) first "
                "or pass optimiser=... into vae.fit(...)."
            )

        if use_amp is None:
            use_amp = self.compile_config.get("use_amp", False)

        if use_adv is None:
            use_adv = self.compile_config.get("use_adv", False)

        if adv_weight is None:
            adv_weight = self.compile_config.get("adv_weight", 0.3)

        if discriminator is None:
            discriminator = self.discriminator

        if disc_optimiser is None:
            disc_optimiser = self.disc_optimiser

        self.to(device)

        if discriminator is not None:
            discriminator.to(device)

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)

        history = self._make_history()

        manager = ModelCheckpoint(
            checkpoint_dir=checkpoint_dir,
            monitor=monitor,
            mode=mode,
            patience=patience,
            min_delta=min_delta,
            save_best_after_epoch=save_best_after_epoch,
            verbose=True,
        )

        start_epoch = self._load_checkpoint_if_possible(
            resume_checkpoint_path=resume_checkpoint_path,
            device=device,
            optimiser=optimiser,
            discriminator=discriminator,
            disc_optimiser=disc_optimiser,
            manager=manager,
        )

        history = self._load_history_if_possible(
            history=history,
            history_path=history_path,
            load_history=load_history,
        )

        scheduler = self._make_scheduler(
            optimiser=optimiser,
            use_scheduler=use_scheduler,
            scheduler_type=scheduler_type,
            mode=mode,
            scheduler_factor=scheduler_factor,
            scheduler_patience=scheduler_patience,
            scheduler_min_lr=scheduler_min_lr,
            n_epochs=n_epochs,
        )

        for epoch in tqdm(range(start_epoch, n_epochs + 1), desc="Training VAE"):
            if use_kl_warmup:
                beta_this_epoch = linear_kl_warmup(
                    epoch=epoch,
                    target_beta=beta_target,
                    warmup_epochs=kl_warmup_epochs,
                    start_beta=beta_start,
                )
            else:
                beta_this_epoch = beta_target

            adv_warmup_done = use_adv and (epoch > adv_warmup_epochs)

            train_metrics = self._train_epoch(
                dataloader=train_loader,
                optimiser=optimiser,
                device=device,
                beta_kl=beta_this_epoch,
                scales=scales,
                grad_clip=grad_clip,
                adv_weight=adv_weight,
                use_amp=use_amp,
                discriminator=discriminator,
                disc_optimiser=disc_optimiser,
                use_adv=use_adv,
                adv_warmup_done=adv_warmup_done,
            )

            val_metrics = self.evaluate(
                dataloader=val_loader,
                device=device,
                beta_kl=beta_this_epoch,
                adv_weight=adv_weight,
                scales=scales,
                use_amp=use_amp,
                discriminator=discriminator,
            )

            if scheduler is not None:
                if scheduler_type == "plateau":
                    scheduler.step(val_metrics["loss"])
                else:
                    scheduler.step()

            current_lr = get_lr(optimiser)

            epoch_record = {
                "epoch": epoch,
                "beta_kl": beta_this_epoch,
                "lr": current_lr,
                "adv_active": int(adv_warmup_done),

                "train_loss": train_metrics["loss"],
                "train_recon_loss": train_metrics["recon_loss"],
                "train_kl_loss": train_metrics["kl_loss"],
                "train_gen_loss": train_metrics["gen_loss"],
                "train_disc_loss": train_metrics["disc_loss"],
                "train_adaptive_weight": train_metrics["adaptive_weight"],
                "train_full_mae": train_metrics["full_mae"],
                "train_full_rmse": train_metrics["full_rmse"],
                "train_psnr": train_metrics["psnr"],

                "val_loss": val_metrics["loss"],
                "val_recon_loss": val_metrics["recon_loss"],
                "val_kl_loss": val_metrics["kl_loss"],
                "val_full_mae": val_metrics["full_mae"],
                "val_full_rmse": val_metrics["full_rmse"],
                "val_psnr": val_metrics["psnr"],

                "val_diagnostic_gen_adv": val_metrics["diagnostic_gen_adv"],
                "val_diagnostic_disc_real": val_metrics["diagnostic_disc_real"],
                "val_diagnostic_disc_fake": val_metrics["diagnostic_disc_fake"],
            }

            for key in history:
                history[key].append(epoch_record[key])

            pd.DataFrame(history).to_csv(history_path, index=False)

            adv_status = "ON" if adv_warmup_done else (
                f"warmup ({epoch}/{adv_warmup_epochs})" if use_adv else "OFF"
            )

            print(f"Epoch {epoch:02d}  | adv: {adv_status}")
            print(f"KL Beta:          {beta_this_epoch:.8f}")
            print(f"Learning Rate:    {current_lr:.8e}")

            print(f"Train Loss:       {train_metrics['loss']:.6f}")
            print(f"Train Recon:      {train_metrics['recon_loss']:.6f}")
            print(f"Train KL:         {train_metrics['kl_loss']:.6f}")

            if use_adv:
                print(f"Train Gen:        {train_metrics['gen_loss']:.6f}")
                print(f"Train Disc:       {train_metrics['disc_loss']:.6f}")
                print(f"Train Adapt W:    {train_metrics['adaptive_weight']:.6f}")

            print(f"Train Full RMSE:  {train_metrics['full_rmse']:.6f}")
            print(f"Train PSNR:       {train_metrics['psnr']:.4f}")

            print(f"Val Loss:         {val_metrics['loss']:.6f}")
            print(f"Val Full RMSE:    {val_metrics['full_rmse']:.6f}")
            print(f"Val PSNR:         {val_metrics['psnr']:.4f}")

            if discriminator is not None:
                print(f"Val Disc Real:    {val_metrics['diagnostic_disc_real']:.6f}  (diagnostic)")
                print(f"Val Disc Fake:    {val_metrics['diagnostic_disc_fake']:.6f}  (diagnostic)")
                print(f"Val Gen Adv:      {val_metrics['diagnostic_gen_adv']:.6f}  (diagnostic)")

            print("-" * 60)

            # keep your external checkpoint manager
            manager.step(
                epoch=epoch,
                metrics=epoch_record,
                model=self,
                optimiser=optimiser,
            )

            full_ckpt = {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "best_score": manager.best_score,
                "best_epoch": manager.best_epoch,
            }

            if discriminator is not None:
                full_ckpt["discriminator_state_dict"] = discriminator.state_dict()
            else:
                full_ckpt["discriminator_state_dict"] = None

            if disc_optimiser is not None:
                full_ckpt["disc_optimiser_state_dict"] = disc_optimiser.state_dict()
            else:
                full_ckpt["disc_optimiser_state_dict"] = None

            torch.save(full_ckpt, checkpoint_dir / "last_state.pt")

            if manager.should_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        return {
            "history": history,
            "best_score": manager.best_score,
            "best_epoch": manager.best_epoch,
            "checkpoint_dir": checkpoint_dir,
        }