import math
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
vae = VAE(in_channels=1, base_channels=32, latent_channels=4).to(device)
--------------------------------------------------------------------------
- Build optimiser:
vae_optimiser = AdamW(vae.parameters(), lr=1e-4, weight_decay=1e-4)
--------------------------------------------------------------------------
- Compile model:
vae.compile(
    optimiser=vae_optimiser,
    device=device,
    recon_lossfn="masked_multires_l1_grad",    
    use_amp=True,
    use_scheduler=True,
    scheduler_type="plateau",          
    scheduler_mode="min",
    scheduler_factor=0.5,
    scheduler_patience=3,
    scheduler_min_lr=1e-7,
    cosine_tmax=50
)
--------------------------------------------------------------------------
- Train model:
vae_results = vae.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    n_epochs=150,

    checkpoint_dir=root_dir / "vae_" / "checkpoints",
    history_path=root_dir / "vae_" / "history" / "history.csv",

    variant="long_gap",                                             # "short_gap" or "long_gap"
    short_recon_lossfn="masked_l1_grad",
    long_recon_lossfn="masked_multires_l1_grad",

    beta_target=1e-4,
    beta_start=0.0,
    use_kl_warmup=True,
    kl_warmup_epochs=10,

    context_weight=0.1,
    grad_weight=0.1,
    delta=1.0,
    scales=(1, 2, 4),
    scale_weights=None,

    monitor="val_gap_rmse",
    mode="min",
    patience=5,
    min_delta=1e-4,
    save_best_after_epoch=1,

    grad_clip=1.0,
    verbose=True,

    resume_checkpoint_path=None,                                   # or path to last_model.pt
    load_history=True,
    resume_scheduler=False,
)
"""

# VAE model class
class VAE(nn.Module):
    # helper blocks
    class ConvGNSiLU(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=8):
            super().__init__()

            # simple conv -> norm -> activation block
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            self.norm = nn.GroupNorm(num_groups=min(groups, out_channels), num_channels=out_channels)
            self.act = nn.SiLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.norm(x)
            x = self.act(x)
            return x
        
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, time_emb_dim=None, groups=8):
            super().__init__()

            self.norm1 = nn.GroupNorm(num_groups=min(groups, in_channels), num_channels=in_channels)
            self.act1 = nn.SiLU()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

            # for reuse with diffusion-style blocks
            self.time_proj = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None

            # second conv branch
            self.norm2 = nn.GroupNorm(num_groups=min(groups, out_channels), num_channels=out_channels)
            self.act2 = nn.SiLU()
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

            # project residual path if channel dimensions change
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        def forward(self, x, t_emb=None):
            residual = self.skip(x)

            h = self.norm1(x)
            h = self.act1(h)
            h = self.conv1(h)
            
            # projected time embed after first conv 
            if self.time_proj is not None and t_emb is not None:
                t_out = self.time_proj(t_emb)
                h = h + t_out[:, :, None, None]

            h = self.norm2(h)
            h = self.act2(h)
            h = self.conv2(h)

            # residual learning to help optimisation and preserve information
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
            # nearest-neighbour upsampling avoids checkerboard artifacts
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

            self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)

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

            # flatten spatial positions so attention is computed over H*W locations
            q = q.view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
            k = k.view(b, self.num_heads, self.head_dim, h * w)
            v = v.view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)

            attn_scores = torch.matmul(q, k) / math.sqrt(self.head_dim)
            attn_weights = torch.softmax(attn_scores, dim=-1)

            out = torch.matmul(attn_weights, v)
            out = out.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
            out = self.proj_out(out)

            return out + residual

    class Encoder(nn.Module):
        def __init__(self, in_channels=1, base_channels=64, latent_channels=4):
            super().__init__()

            RB = VAE.ResidualBlock
            DS = VAE.DownSample
            SA = VAE.SelfAttention

            self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

            self.block1 = nn.Sequential(
                RB(base_channels, base_channels),
                RB(base_channels, base_channels),
            )
            self.down1 = DS(base_channels)

            self.block2 = nn.Sequential(
                RB(base_channels, base_channels * 2),
                RB(base_channels * 2, base_channels * 2),
            )
            self.down2 = DS(base_channels * 2)

            self.block3 = nn.Sequential(
                RB(base_channels * 2, base_channels * 4),
                RB(base_channels * 4, base_channels * 4),
            )
            self.down3 = DS(base_channels * 4)

            self.block4 = nn.Sequential(
                RB(base_channels * 4, base_channels * 4),
                RB(base_channels * 4, base_channels * 4),
            )
            self.down4 = DS(base_channels * 4)

            self.mid = nn.Sequential(
                RB(base_channels * 4, base_channels * 4),
                SA(base_channels * 4, num_heads=4),
                RB(base_channels * 4, base_channels * 4),
            )

            # posterior parameters q(z|x)
            self.mu_head = nn.Conv2d(base_channels * 4, latent_channels, kernel_size=1)
            self.logvar_head = nn.Conv2d(base_channels * 4, latent_channels, kernel_size=1)

        def forward(self, x):
            x = self.in_conv(x)

            x = self.block1(x)
            x = self.down1(x)

            x = self.block2(x)
            x = self.down2(x)

            x = self.block3(x)
            x = self.down3(x)

            x = self.block4(x)
            x = self.down4(x)

            x = self.mid(x)

            mu = self.mu_head(x)
            logvar = self.logvar_head(x)
            return mu, logvar

    class Decoder(nn.Module):
        def __init__(self, out_channels=1, base_channels=64, latent_channels=4):
            super().__init__()

            RB = VAE.ResidualBlock
            US = VAE.UpSample
            SA = VAE.SelfAttention

            self.in_conv = nn.Conv2d(latent_channels, base_channels * 4, kernel_size=3, padding=1)

            self.mid = nn.Sequential(
                RB(base_channels * 4, base_channels * 4),
                SA(base_channels * 4, num_heads=4),
                RB(base_channels * 4, base_channels * 4),
            )

            self.block4 = nn.Sequential(
                RB(base_channels * 4, base_channels * 4),
                RB(base_channels * 4, base_channels * 4),
            )
            self.up4 = US(base_channels * 4)

            self.block3 = nn.Sequential(
                RB(base_channels * 4, base_channels * 4),
                RB(base_channels * 4, base_channels * 2),
            )
            self.up3 = US(base_channels * 2)

            self.block2 = nn.Sequential(
                RB(base_channels * 2, base_channels * 2),
                RB(base_channels * 2, base_channels),
            )
            self.up2 = US(base_channels)

            self.block1 = nn.Sequential(
                RB(base_channels, base_channels),
                RB(base_channels, base_channels),
            )
            self.up1 = US(base_channels)

            self.out_norm = nn.GroupNorm(num_groups=min(8, base_channels), num_channels=base_channels)
            self.out_act = nn.SiLU()
            self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

        def forward(self, z):
            x = self.in_conv(z)
            x = self.mid(x)

            x = self.block4(x)
            x = self.up4(x)

            x = self.block3(x)
            x = self.up3(x)

            x = self.block2(x)
            x = self.up2(x)

            x = self.block1(x)
            x = self.up1(x)

            x = self.out_norm(x)
            x = self.out_act(x)
            x = self.out_conv(x)
            return x

    # main VAE builder
    def __init__(self, in_channels=1, base_channels=64, latent_channels=4):
        super().__init__()

        self.encoder = self.Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
        )
        self.decoder = self.Decoder(
            out_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
        )

        self.latent_channels = latent_channels

        # filled in later by compile()
        self.optimiser = None
        self.scheduler = None
        self.device_obj = None
        self.device_type = "cpu"
        self.use_amp = False
        self.scaler = None
        self.recon_lossfn = None
        self.scheduler_type = None

    # helpers utility functions
    @staticmethod
    def padding(x, multiple=16):
        # VAE downsamples 4 times, so H and W should be divisible by 16
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
    def get_loss_name(variant="long_gap", shortgap_loss="masked_l1_grad", longgap_loss="masked_multires_l1_grad"):
        if variant == "short_gap":
            return shortgap_loss
        elif variant == "long_gap":
            return longgap_loss
        else:
            raise ValueError(f"Invalid gap variant {variant}.")

    @staticmethod
    def linear_kl_warmup(epoch, target_beta=1e-4, warmup_epochs=10, start_beta=0.0):
        if warmup_epochs <= 0:
            return target_beta
        progress = min(epoch / warmup_epochs, 1.0)
        beta = start_beta + progress * (target_beta - start_beta)
        return beta

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterise(self, mu, logvar):
        # reparameterisation trick makes sampling differentiable
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

    # losses configurations
    def compute_reconloss(
        self,
        recon,
        target,
        mask,
        lossfn="masked_l1_grad",
        context_weight=0.1,
        grad_weight=0.1,
        delta=1.0,
        scales=(1, 2, 4),
        scale_weights=None,
        eps=1e-8,
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
                eps=eps,
            )

        elif lossfn == "masked_multires_l1":
            total_loss, gap_loss, context_loss = masked_multires_l1_loss(
                pred=recon,
                target=target,
                mask=mask,
                context_weight=context_weight,
                scales=scales,
                scale_weights=scale_weights,
                eps=eps,
            )
            grad_loss = torch.tensor(0.0, device=recon.device)

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
        beta_kl=1e-4,
        recon_lossfn="masked_l1_grad",
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

        # KL divergence push the latent posterior toward a standard Gaussian
        kl_per_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
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

    def compute_metrics(self, recon_eval, y_eval, m_eval):
        return {
            "gap_mae": masked_mae(recon_eval, y_eval, m_eval).item(),
            "gap_rmse": masked_rmse(recon_eval, y_eval, m_eval).item(),
            "full_mae": full_mae(recon_eval, y_eval).item(),
            "full_rmse": full_rmse(recon_eval, y_eval).item(),
            "psnr": psnr(recon_eval, y_eval).item(),
        }

    def running_dict(self):
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

    # compile function
    def compile(
        self,
        optimiser,
        device,
        recon_lossfn="masked_l1_grad",
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
        self.recon_lossfn = recon_lossfn

        self.device_obj = device if isinstance(device, torch.device) else torch.device(device)
        self.device_type = self.device_obj.type
        self.to(self.device_obj)

        # AMP is enabled only on CUDA
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

    # train for one epochs
    def train_epoch(
        self,
        dataloader,
        device,
        beta_kl=1e-4,
        context_weight=0.1,
        grad_weight=0.1,
        delta=1.0,
        scales=(1, 2, 4),
        scale_weights=None,
        grad_clip=1.0,
    ):
        self.train()
        running = self.running_dict()
        n_batches = 0

        for batch in tqdm(dataloader, desc="Training VAE", leave=False):
            y = batch["y"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)

            y, pad_info = self.padding(y, multiple=16)
            m, _ = self.padding(m, multiple=16)

            self.optimiser.zero_grad(set_to_none=True)

            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                recon, mu, logvar, z = self(y)

                loss_dict = self.vae_loss(
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

            self.scaler.scale(loss_dict["loss"]).backward()

            if grad_clip is not None:
                self.scaler.unscale_(self.optimiser)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip)

            self.scaler.step(self.optimiser)
            self.scaler.update()

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

    # evaluation function for a single epoch
    @torch.no_grad()
    def evaluate_epoch(
        self,
        dataloader,
        device,
        beta_kl=1e-4,
        context_weight=0.1,
        grad_weight=0.1,
        delta=1.0,
        scales=(1, 2, 4),
        scale_weights=None,
    ):
        self.eval()
        running = self.running_dict()
        n_batches = 0

        for batch in tqdm(dataloader, desc="Eval VAE", leave=False):
            y = batch["y"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)

            y, pad_info = self.padding(y, multiple=16)
            m, _ = self.padding(m, multiple=16)

            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                recon, mu, logvar, z = self(y)

                loss_dict = self.vae_loss(
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

    # fit
    def fit(
        self,
        train_loader,
        val_loader,
        device,
        n_epochs,
        checkpoint_dir,
        history_path,
        variant="long_gap",
        short_recon_lossfn="masked_l1_grad",
        long_recon_lossfn="masked_multires_l1_grad",
        beta_target=1e-4,
        beta_start=0.0,
        use_kl_warmup=True,
        kl_warmup_epochs=10,
        context_weight=0.1,
        grad_weight=0.1,
        delta=1.0,
        scales=(1, 2, 4),
        scale_weights=None,
        monitor="val_gap_rmse",
        mode="min",
        patience=10,
        min_delta=1e-4,
        save_best_after_epoch=5,
        grad_clip=1.0,
        verbose=True,
        resume_checkpoint_path=None,
        load_history=True,
        resume_scheduler=False,
    ):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)

        self.recon_lossfn = self.get_loss_name(
            variant=variant,
            shortgap_loss=short_recon_lossfn,
            longgap_loss=long_recon_lossfn,
        )
        if verbose:
            print(f"Using VAE reconstruction loss: {self.recon_lossfn} for gap {variant}")

        self.history = {
            "epoch": [],
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

            train_metrics = self.train_epoch(
                dataloader=train_loader,
                device=device,
                beta_kl=beta_this_epoch,
                context_weight=context_weight,
                grad_weight=grad_weight,
                delta=delta,
                scales=scales,
                scale_weights=scale_weights,
                grad_clip=grad_clip,
            )

            val_metrics = self.evaluate_epoch(
                dataloader=val_loader,
                device=device,
                beta_kl=beta_this_epoch,
                context_weight=context_weight,
                grad_weight=grad_weight,
                delta=delta,
                scales=scales,
                scale_weights=scale_weights,
            )

            if self.scheduler is not None:
                if self.scheduler_type == "plateau":
                    if monitor == "val_gap_rmse":
                        self.scheduler.step(val_metrics["gap_rmse"])
                    elif monitor == "val_loss":
                        self.scheduler.step(val_metrics["loss"])
                    else:
                        self.scheduler.step(val_metrics["loss"])
                elif self.scheduler_type == "cosine":
                    self.scheduler.step()

            current_lr = self.get_lr()

            epoch_record = {
                "epoch": epoch,
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
            }

            for key in self.history:
                self.history[key].append(epoch_record[key])

            pd.DataFrame(self.history).to_csv(history_path, index=False)

            if verbose:
                print(f"Epoch {epoch:02d}")
                print(f"KL Beta:          {beta_this_epoch:.8f}")
                print(f"Learning Rate:    {current_lr:.8e}")
                print(f"Train Loss:       {train_metrics['loss']:.6f}")
                print(f"Val Loss:         {val_metrics['loss']:.6f}")
                print(f"Train Recon Loss: {train_metrics['recon_loss']:.6f}")
                print(f"Val Recon Loss:   {val_metrics['recon_loss']:.6f}")
                print(f"Train KL Loss:    {train_metrics['kl_loss']:.6f}")
                print(f"Val KL Loss:      {val_metrics['kl_loss']:.6f}")
                print(f"Train Gap RMSE:   {train_metrics['gap_rmse']:.6f}")
                print(f"Val Gap RMSE:     {val_metrics['gap_rmse']:.6f}")
                print(f"Train PSNR:       {train_metrics['psnr']:.4f}")
                print(f"Val PSNR:         {val_metrics['psnr']:.4f}")
                print("-" * 60)

            # best/last saving and early stopping to ModelCheckpoint
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
            "recon_lossfn": self.recon_lossfn,
        }