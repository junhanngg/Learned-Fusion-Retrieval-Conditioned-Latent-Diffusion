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

"""
Usage: 
--------------------------------------------------------------------------
- Create Model:
vae = VAE(
    in_channels=1, 
    base_channels=32, 
    latent_channels=8
    ).to(device)
--------------------------------------------------------------------------
- Build optimiser:
vae_optimiser = AdamW(vae.parameters(), lr=1e-4, weight_decay=1e-4)
--------------------------------------------------------------------------
- Directories:
vae_checkpoint_dir = root_dir / "vae_" / "checkpoints"
vae_history_dir = root_dir / "vae_" / "history"

vae_checkpoint_dir.mkdir(parents=True, exist_ok=True)
vae_history_dir.mkdir(parents=True, exist_ok=True)
--------------------------------------------------------------------------
- Compile model:
vae.compile(
    optimiser=vae_optimiser,
    device=device,
    use_scheduler=True,
    scheduler_type="plateau",
    scheduler_factor=0.3,
    scheduler_patience=4,
    scheduler_min_lr=1e-7,
    scheduler_mode="min",
    use_amp=True,
)
--------------------------------------------------------------------------
- Train model:
vae_results = vae.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    n_epochs=150,
    variant="long_gap",
    checkpoint_dir=vae_checkpoint_dir,
    history_path=vae_history_dir / "history.csv",

    short_recon_lossfn="masked_l1_grad",
    long_recon_lossfn="masked_multires_l1_grad",

    beta_target=3e-4,
    beta_start=0.0,
    use_kl_warmup=True,
    kl_warmup_epochs=10,

    masked_input_start=0.05,
    masked_input_end=0.3,
    masked_input_ramp_end=25,

    val_masked_input_prob=0.2,

    context_weight=0.1,
    grad_weight=0.1,
    delta=1.0,
    scales=(1, 2, 4),
    scale_weights=None,

    monitor="val_loss",
    mode="min",
    patience=8,
    min_delta=1e-4,
    save_best_after_epoch=1,

    grad_clip=1.0,
    resume_checkpoint_path=vae_checkpoint_dir / "last_model.pt",
    load_history=True,

    use_amp=True,
)
"""
class VAE(nn.Module):
    # helpers
    @staticmethod
    def make_norm(channels, max_groups=8):
        groups = min(max_groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)

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
    def get_lr(optimiser):
        return optimiser.param_groups[0]["lr"]

    @staticmethod
    def linear_kl_warmup(epoch, target_beta=5e-4, warmup_epochs=15, start_beta=0.0):
        if warmup_epochs <= 0:
            return target_beta

        progress = min(epoch / warmup_epochs, 1.0)
        beta = start_beta + progress * (target_beta - start_beta)
        return beta

    @staticmethod
    def masked_input_scheduler(epoch, start_prob=0.1, end_prob=0.8, ramp_end_epoch=50):
        if epoch >= ramp_end_epoch:
            return end_prob

        progress = (epoch - 1) / max(ramp_end_epoch - 1, 1)
        prob = start_prob + progress * (end_prob - start_prob)
        return prob

    @staticmethod
    def make_running_dict():
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

    @staticmethod
    def get_loss(
        variant="long_gap",
        shortgap_loss="masked_l1_grad",
        longgap_loss="masked_multires_l1_grad",
    ):
        if variant == "short_gap":
            loss = shortgap_loss
        elif variant == "long_gap":
            loss = longgap_loss
        else:
            raise ValueError(f"Invalid gap variant {variant}.")
        return loss

    @staticmethod
    def choose_input(batch, device, masked_input_prob=0.7):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)

        use_masked = torch.rand(1).item() < masked_input_prob
        encoder_input = x if use_masked else y

        return encoder_input, y, m

    # building blocks
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, dropout=0.0):
            super().__init__()

            self.norm1 = VAE.make_norm(in_channels)
            self.act1 = nn.SiLU()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

            self.norm2 = VAE.make_norm(out_channels)
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

            self.norm1 = VAE.make_norm(channels)
            self.act1 = nn.SiLU()
            self.conv1 = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            )

            self.norm2 = VAE.make_norm(channels)
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

            self.norm = VAE.make_norm(channels)

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

    class Encoder(nn.Module):
        def __init__(self, in_channels=1, base_channels=64, latent_channels=8):
            super().__init__()

            c1 = base_channels
            c2 = base_channels * 2
            c3 = base_channels * 4
            c4 = base_channels * 4

            self.in_conv = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

            self.block1 = nn.Sequential(
                VAE.ResidualBlock(c1, c1),
                VAE.ResidualBlock(c1, c1)
            )
            self.down1 = VAE.DownSample(c1)

            self.block2 = nn.Sequential(
                VAE.ResidualBlock(c1, c2),
                VAE.ResidualBlock(c2, c2)
            )

            self.block3 = nn.Sequential(
                VAE.ResidualBlock(c2, c2),
                VAE.ResidualBlock(c2, c2)
            )
            self.down2 = VAE.DownSample(c2)

            self.stage4 = nn.Sequential(
                VAE.ResidualBlock(c2, c3),
                VAE.ResidualBlock(c3, c3),
            )

            self.stage5 = nn.Sequential(
                VAE.ResidualBlock(c3, c3),
                VAE.ResidualBlock(c3, c3),
            )
            self.down3 = VAE.DownSample(c3)

            self.stage6 = nn.Sequential(
                VAE.ResidualBlock(c3, c4),
                VAE.ResidualBlock(c4, c4),
            )

            self.mid = nn.Sequential(
                VAE.ResidualBlock(c4, c4),
                VAE.DilatedResBlock(c4, dilation=2),
                VAE.SelfAttention(c4, num_heads=4),
                VAE.DilatedResBlock(c4, dilation=4),
                VAE.ResidualBlock(c4, c4),
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
        def __init__(self, out_channels=1, base_channels=64, latent_channels=8):
            super().__init__()

            c1 = base_channels
            c2 = base_channels * 2
            c3 = base_channels * 4
            c4 = base_channels * 4

            self.in_conv = nn.Conv2d(latent_channels, c4, kernel_size=3, padding=1)

            self.mid = nn.Sequential(
                VAE.ResidualBlock(c4, c4),
                VAE.SelfAttention(c4, num_heads=4),
                VAE.DilatedResBlock(c4, dilation=2),
                VAE.ResidualBlock(c4, c4),
            )

            self.stage6 = nn.Sequential(
                VAE.ResidualBlock(c4, c4),
                VAE.ResidualBlock(c4, c4),
            )
            self.up3 = VAE.UpSample(c4)

            self.stage5 = nn.Sequential(
                VAE.ResidualBlock(c4, c3),
                VAE.ResidualBlock(c3, c3),
            )
            self.stage4 = nn.Sequential(
                VAE.ResidualBlock(c3, c3),
                VAE.ResidualBlock(c3, c3),
            )
            self.up2 = VAE.UpSample(c3)

            self.stage3 = nn.Sequential(
                VAE.ResidualBlock(c3, c2),
                VAE.ResidualBlock(c2, c2),
            )
            self.stage2 = nn.Sequential(
                VAE.ResidualBlock(c2, c2),
                VAE.ResidualBlock(c2, c2),
            )
            self.up1 = VAE.UpSample(c2)

            self.stage1 = nn.Sequential(
                VAE.ResidualBlock(c2, c1),
                VAE.ResidualBlock(c1, c1),
            )

            self.out_norm = VAE.make_norm(c1)
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

    # model init
    def __init__(self, in_channels=1, base_channels=64, latent_channels=8):
        super().__init__()

        self.encoder = self.Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels
        )

        self.decoder = self.Decoder(
            out_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels
        )

        self.latent_channels = latent_channels
        self.in_channels = in_channels
        self.base_channels = base_channels

        # compile-time state
        self.optimiser = None
        self.device_ = None
        self.scheduler = None
        self.scheduler_type = None
        self.use_amp = False
        self.compiled = False
        self.history = None

    # forward
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    # losses
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
        eps=1e-8
    ):
        if lossfn == "masked_l1":
            total_loss, gap_loss, context_loss = masked_l1_loss(
                pred=recon,
                target=target,
                mask=mask,
                context_weight=context_weight,
                eps=eps,
            )
            grad_loss = torch.tensor(0.0, device=recon.device)

        elif lossfn == "masked_huber":
            total_loss, gap_loss, context_loss = masked_huber_loss(
                pred=recon,
                target=target,
                mask=mask,
                context_weight=context_weight,
                delta=delta,
                eps=eps,
            )
            grad_loss = torch.tensor(0.0, device=recon.device)

        elif lossfn == "masked_l1_grad":
            total_loss, gap_loss, context_loss, grad_loss = masked_l1_grad_loss(
                pred=recon,
                target=target,
                mask=mask,
                context_weight=context_weight,
                grad_weight=grad_weight,
                eps=eps
            )

        elif lossfn == "masked_huber_grad":
            total_loss, gap_loss, context_loss, grad_loss = masked_huber_grad_loss(
                pred=recon,
                target=target,
                mask=mask,
                context_weight=context_weight,
                grad_weight=grad_weight,
                delta=delta,
                eps=eps
            )

        elif lossfn == "masked_multires_l1":
            total_loss, gap_loss, context_loss = masked_multires_l1_loss(
                pred=recon,
                target=target,
                mask=mask,
                context_weight=context_weight,
                scales=scales,
                scale_weights=scale_weights,
                eps=eps
            )
            grad_loss = torch.tensor(0.0, device=recon.device)

        elif lossfn == "masked_multires_l1_grad":
            total_loss, gap_loss, context_loss, grad_loss = masked_multires_l1_grad_loss(
                pred=recon,
                target=target,
                mask=mask,
                context_weight=context_weight,
                grad_weight=grad_weight,
                scales=scales,
                scale_weights=scale_weights,
                eps=eps
            )

        else:
            raise ValueError(f"Unsupported VAE reconstruction loss: {lossfn}")

        return {
            "recon_loss": total_loss,
            "gap_loss": gap_loss.detach(),
            "context_loss": context_loss.detach(),
            "grad_loss": grad_loss.detach(),
        }

    def vae_loss(
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
        eps=1e-8
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
            eps=eps
        )

        mu32 = mu.float()
        logvar32 = logvar.float()
        logvar32 = torch.clamp(logvar32, min=-10.0, max=10.0)

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
    def compute_metrics(self, recon_eval, y_eval, m_eval):
        return {
            "gap_mae": masked_mae(recon_eval, y_eval, m_eval).item(),
            "gap_rmse": masked_rmse(recon_eval, y_eval, m_eval).item(),
            "full_mae": full_mae(recon_eval, y_eval).item(),
            "full_rmse": full_rmse(recon_eval, y_eval).item(),
            "psnr": psnr(recon_eval, y_eval).item(),
        }

    # compile
    def compile(
        self,
        optimiser,
        device,
        use_scheduler=True,
        scheduler_type="plateau",
        scheduler_factor=0.5,
        scheduler_patience=3,
        scheduler_min_lr=1e-7,
        scheduler_mode="min",
        cosine_tmax=None,
        use_amp=False,
    ):
        self.optimiser = optimiser
        self.device_ = device
        self.use_amp = use_amp
        self.scheduler_type = scheduler_type
        self.scheduler = None

        if use_scheduler:
            if scheduler_type == "plateau":
                self.scheduler = ReduceLROnPlateau(
                    optimiser,
                    mode=scheduler_mode,
                    factor=scheduler_factor,
                    patience=scheduler_patience,
                    min_lr=scheduler_min_lr,
                )
            elif scheduler_type == "cosine":
                if cosine_tmax is None:
                    raise ValueError("cosine_tmax must be provided when scheduler_type='cosine'")
                self.scheduler = CosineAnnealingLR(
                    optimiser,
                    T_max=cosine_tmax,
                    eta_min=scheduler_min_lr,
                )
            else:
                raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

        self.compiled = True
        return self

    # one epoch train
    def train_epoch(
        self,
        dataloader,
        beta_kl=5e-4,
        recon_lossfn="masked_l1_grad",
        masked_input_prob=0.7,
        context_weight=0.1,
        grad_weight=0.1,
        delta=1.0,
        scales=(1, 2, 4),
        scale_weights=None,
        grad_clip=1.0,
        use_amp=True
    ):
        super().train()

        running = self.make_running_dict()
        n_batches = 0

        use_amp = use_amp and ("cuda" in str(self.device_))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for batch in tqdm(dataloader, desc="Training VAE", leave=False):
            x_in, y, m = self.choose_input(
                batch=batch,
                device=self.device_,
                masked_input_prob=masked_input_prob
            )

            x_in, _ = self.padding(x_in, multiple=8)
            y, pad_info = self.padding(y, multiple=8)
            m, _ = self.padding(m, multiple=8)

            self.optimiser.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                recon, mu, logvar, z = self(x_in)

                loss_dict = self.vae_loss(
                    recon=recon,
                    target=y,
                    mu=mu,
                    logvar=logvar,
                    mask=m,
                    beta_kl=beta_kl,
                    recon_lossfn=recon_lossfn,
                    context_weight=context_weight,
                    grad_weight=grad_weight,
                    delta=delta,
                    scales=scales,
                    scale_weights=scale_weights,
                )

                if not torch.isfinite(loss_dict["loss"]):
                    print("Non-finite loss detected")
                    print("mu min/max/mean:", mu.min().item(), mu.max().item(), mu.mean().item())
                    print("logvar min/max/mean:", logvar.min().item(), logvar.max().item(), logvar.mean().item())
                    raise ValueError("Loss became non-finite")

            scaler.scale(loss_dict["loss"]).backward()

            if grad_clip is not None:
                scaler.unscale_(self.optimiser)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip)

            scaler.step(self.optimiser)
            scaler.update()

            recon_eval = self.unpadding(recon.detach(), pad_info)
            y_eval = self.unpadding(y, pad_info)
            m_eval = self.unpadding(m, pad_info)

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

    # eval
    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        beta_kl=5e-4,
        recon_lossfn="masked_multires_l1_grad",
        masked_input_prob=1.0,
        context_weight=0.1,
        grad_weight=0.1,
        delta=1.0,
        scales=(1, 2, 4),
        scale_weights=None,
        use_amp=True,
        clean=False,
    ):
        super().eval()

        running = self.make_running_dict()
        n_batches = 0

        use_amp = use_amp and ("cuda" in str(self.device_))
        desc = "Eval VAE Clean" if clean else "Eval VAE"

        for batch in tqdm(dataloader, desc=desc, leave=False):
            if clean:
                y = batch["y"].to(self.device_, non_blocking=True)
                m = batch["mask"].to(self.device_, non_blocking=True)

                x_in, _ = self.padding(y, multiple=8)
                y, pad_info = self.padding(y, multiple=8)
                m, _ = self.padding(m, multiple=8)
            else:
                x_in, y, m = self.choose_input(
                    batch=batch,
                    device=self.device_,
                    masked_input_prob=masked_input_prob,
                )

                x_in, _ = self.padding(x_in, multiple=8)
                y, pad_info = self.padding(y, multiple=8)
                m, _ = self.padding(m, multiple=8)

            with torch.amp.autocast("cuda", enabled=use_amp):
                recon, mu, logvar, z = self(x_in)

                loss_dict = self.vae_loss(
                    recon=recon,
                    target=y,
                    mu=mu,
                    logvar=logvar,
                    mask=m,
                    beta_kl=beta_kl,
                    recon_lossfn=recon_lossfn,
                    context_weight=context_weight,
                    grad_weight=grad_weight,
                    delta=delta,
                    scales=scales,
                    scale_weights=scale_weights,
                )

            recon_eval = self.unpadding(recon.detach(), pad_info)
            y_eval = self.unpadding(y, pad_info)
            m_eval = self.unpadding(m, pad_info)

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

    # reconstruction helper
    @torch.no_grad()
    def reconstruct(self, x, sample=False):
        super().eval()

        x_pad, pad_info = self.padding(x, multiple=8)
        mu, logvar = self.encode(x_pad)

        if sample:
            z = self.reparameterise(mu, logvar)
        else:
            z = mu

        recon = self.decode(z)
        recon = self.unpadding(recon, pad_info)
        return recon

    # checkpoint loading helper for resume
    def load_checkpoint(self, checkpoint_path, map_location=None, load_optimiser=True):
        ckpt = torch.load(checkpoint_path, map_location=map_location)

        self.load_state_dict(ckpt["model_state_dict"])

        if load_optimiser and self.optimiser is not None and "optimiser_state_dict" in ckpt:
            self.optimiser.load_state_dict(ckpt["optimiser_state_dict"])

        return ckpt

    # full fit loop
    def fit(
        self,
        train_loader,
        val_loader,
        n_epochs,
        variant,
        checkpoint_dir,
        history_path,
        short_recon_lossfn="masked_l1_grad",
        long_recon_lossfn="masked_multires_l1_grad",
        beta_target=5e-4,
        beta_start=0.0,
        use_kl_warmup=True,
        kl_warmup_epochs=15,
        masked_input_start=0.1,
        masked_input_end=0.8,
        masked_input_ramp_end=50,
        val_masked_input_prob=0.5,
        context_weight=0.1,
        grad_weight=0.1,
        delta=1.0,
        scales=(1, 2, 4),
        scale_weights=None,
        monitor="val_gap_rmse",
        mode="min",
        patience=8,
        min_delta=1e-4,
        save_best_after_epoch=3,
        grad_clip=1.0,
        resume_checkpoint_path=None,
        load_history=True,
        use_amp=False
    ):
        if not self.compiled:
            raise RuntimeError("Call model.compile(...) before model.fit(...).")

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)

        recon_lossfn = self.get_loss(
            variant=variant,
            shortgap_loss=short_recon_lossfn,
            longgap_loss=long_recon_lossfn,
        )
        print(f"Using VAE reconstruction loss: {recon_lossfn} for gap {variant}")

        history = {
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
            verbose=True,
        )

        start_epoch = 1

        if resume_checkpoint_path is not None:
            resume_checkpoint_path = Path(resume_checkpoint_path)
            if resume_checkpoint_path.exists():
                ckpt = self.load_checkpoint(
                    resume_checkpoint_path,
                    map_location=self.device_,
                    load_optimiser=True,
                )

                start_epoch = ckpt.get("epoch", 0) + 1
                manager.best_score = ckpt.get("best_score", manager.best_score)
                manager.best_epoch = ckpt.get("best_epoch", manager.best_epoch)

                print(f"Resumed from: {resume_checkpoint_path}")
                print(f"Continuing from epoch: {start_epoch}")

        if load_history and history_path.exists():
            old_history_df = pd.read_csv(history_path)
            missing_cols = [k for k in history.keys() if k not in old_history_df.columns]
            if len(missing_cols) == 0:
                history = {col: old_history_df[col].tolist() for col in old_history_df.columns}
                print(f"Loaded existing history from: {history_path}")

        for epoch in tqdm(range(start_epoch, n_epochs + 1), desc="Training VAE"):
            if use_kl_warmup:
                beta_this_epoch = self.linear_kl_warmup(
                    epoch=epoch,
                    target_beta=beta_target,
                    warmup_epochs=kl_warmup_epochs,
                    start_beta=beta_start,
                )
            else:
                beta_this_epoch = beta_target

            masked_input_prob = self.masked_input_scheduler(
                epoch=epoch,
                start_prob=masked_input_start,
                end_prob=masked_input_end,
                ramp_end_epoch=masked_input_ramp_end
            )

            train_metrics = self.train_epoch(
                dataloader=train_loader,
                beta_kl=beta_this_epoch,
                recon_lossfn=recon_lossfn,
                masked_input_prob=masked_input_prob,
                context_weight=context_weight,
                grad_weight=grad_weight,
                delta=delta,
                scales=scales,
                scale_weights=scale_weights,
                grad_clip=grad_clip,
                use_amp=use_amp
            )

            val_metrics = self.evaluate(
                dataloader=val_loader,
                beta_kl=beta_this_epoch,
                recon_lossfn=recon_lossfn,
                masked_input_prob=val_masked_input_prob,
                context_weight=context_weight,
                grad_weight=grad_weight,
                delta=delta,
                scales=scales,
                scale_weights=scale_weights,
                use_amp=use_amp,
                clean=False,
            )

            clean_val_metrics = self.evaluate(
                dataloader=val_loader,
                beta_kl=beta_this_epoch,
                recon_lossfn=recon_lossfn,
                context_weight=context_weight,
                grad_weight=grad_weight,
                delta=delta,
                scales=scales,
                scale_weights=scale_weights,
                use_amp=use_amp,
                clean=True,
            )

            if self.scheduler is not None:
                if self.scheduler_type == "plateau":
                    if monitor == "val_gap_rmse":
                        self.scheduler.step(val_metrics["gap_rmse"])
                    else:
                        self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            current_lr = self.get_lr(self.optimiser)

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

            for key in history:
                history[key].append(epoch_record[key])

            pd.DataFrame(history).to_csv(history_path, index=False)

            print(f"Epoch {epoch:02d}")
            print(f"Masked Input Prob: {masked_input_prob:.3f}")
            print(f"KL Beta:          {beta_this_epoch:.8f}")
            print(f"Learning Rate:    {current_lr:.8e}")

            print(f"Train Loss:       {train_metrics['loss']:.6f}")
            print(f"Train Gap RMSE:   {train_metrics['gap_rmse']:.6f}")
            print(f"Train PSNR:       {train_metrics['psnr']:.4f}")

            print(f"Masked Val Loss:      {val_metrics['loss']:.6f}")
            print(f"Masked Val Gap RMSE:  {val_metrics['gap_rmse']:.6f}")
            print(f"Masked Val PSNR:      {val_metrics['psnr']:.4f}")

            print(f"Clean Val Loss:       {clean_val_metrics['loss']:.6f}")
            print(f"Clean Val Gap RMSE:   {clean_val_metrics['gap_rmse']:.6f}")
            print(f"Clean Val PSNR:       {clean_val_metrics['psnr']:.4f}")
            print("-" * 60)

            manager.step(
                epoch=epoch,
                metrics=epoch_record,
                model=self,
                optimiser=self.optimiser,
            )

            if manager.should_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        self.history = history

        return {
            "history": history,
            "best_score": manager.best_score,
            "best_epoch": manager.best_epoch,
            "checkpoint_dir": checkpoint_dir,
            "recon_lossfn": recon_lossfn,
        }