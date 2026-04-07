# import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
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
    gap_seconds=0.5,            
    lossfn="masked_l1",                     # defaulted to masked_l1 if not defined
    shortgap_loss="masked_l1",              # for 0.5 and 2.0
    longgap_loss="masked_multires_l1",      # for 3.0 and 5.0
    lr=1e-3,                                # learning rate
    weight_decay=1e-5,                      # lr decay steps
    grad_clip=1.0,                         
).to(device)

--------------------------------------------------------------------------
- Creating optimiser:

model.configure_optimiser()

--------------------------------------------------------------------------
- Training model:

results = model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    n_epochs=100,                           # recommended epoch training
    checkpoint_dir=checkpoint_dir,          # hardcoded in script (see below)
    history_csv_path=history_dir,           # hardcoded in script (see below)
    monitor="val_gap_rmse",                 # early stopping/best model monitored on this
    mode="min",
    patience=5,                             # patience before early stoppping
    min_delta=1e-4,             
    save_best_after_epoch=10                 # num of epochs before checking to save
)
--------------------------------------------------------------------------

- Inferencing:
batch = next(iter(test_loader))       
x = batch["x"][0]
y = batch["y"][0]
mask = batch["mask"][0]

pred = model.inference(x, mask)                # generates prediction for gaps
full_pred = model.reconstruct_full(x, mask)    # construct the full spectrogram

"""

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# define checkpoints and history 
root_dir = Path("training_logs/CNN_AE")
# root_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir = root_dir / "model_weights"
history_dir = root_dir / "history_csv"

# squeeze and excitation block to learn channel-wise importance weights
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()

        # reduce channel dimension in the small gating network
        hidden = max(channels // reduction, 8)

        # global average pooling: converts each channel from (H, W) into a single summary value
        self.pool = nn.AdaptiveAvgPool2d(1) # (B, C, 1, 1)

        # first 1x1 convolution: reduces the number of channels to a smaller hidden size
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)

        # second 1x1 convolution: maps the hidden representation back to the original number of channels
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    # forward pass
    def forward(self, x):
        #squeeze spatial information into one value per channel
        scale = self.pool(x)

        # small nonlinear transformation
        scale = F.silu(self.fc1(scale))

        # produce channel weights between 0 and 1
        scale = torch.sigmoid(self.fc2(scale))

        # reweight original feature map channel-wise
        return x * scale

# efficientnet MBConvBlock, expands channels with 1 by 1 convolution and apply depthwise convolution then apply SE attention
# and projects back to desired output channels, later on it adds aresidual connection
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, kernel_size=3):
        super().__init__()

        # expanded channel dimension inside the block
        hidden_dim = in_channels * expansion

        # use residual only when input and output channle match
        self.use_residual = (in_channels == out_channels)

        # 1x1 expansion convolution: increases channel capacity 
        self.expand = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        # depthwise convolution: applies one spatial filter per channel separately
        # groups = hidden_dim means each channel is convolved independently
        self.depthwise = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        # SE attention block
        self.se = SEBlock(hidden_dim)

        # 1x1 projection convolution: compresses channels back down to out_channels.
        self.project = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # expand channels
        out = self.expand(x)
        out = F.silu(self.bn1(out))

        # learn spatial patterns channel-by-channel
        out = self.depthwise(out)
        out = F.silu(self.bn2(out))

        # channel attention
        out = self.se(out)

        # project back to output channel size
        out = self.project(out)
        out = self.bn3(out)

        # add skip connection
        if self.use_residual:
            out = out + x

        # final nonlinearity
        return F.silu(out)

# CNN autoencoder class
class CNNAutoencoder(nn.Module):
    def __init__(
        self,
        gap_seconds=0.5,
        lossfn="masked_l1",
        shortgap_loss="masked_l1",
        longgap_loss="masked_multires_l1",
        lr=1e-3,
        weight_decay=1e-5,
        grad_clip=1.0,
        device=None,
    ):
        super().__init__()

        self.gap_seconds = gap_seconds
        self.shortgap_loss = shortgap_loss
        self.longgap_loss = longgap_loss
        self.lossfn = self.get_loss(lossfn)

        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.device_name = device

        # metwork architecture
        # stem layer that combines the 2-channel input into an initial feature representation
        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        # encoder layer

        # 1st encoder stage, keeps same spatial size, refines features
        self.enc1 = MBConvBlock(32, 32, expansion=2, kernel_size=3)

        # downsampel 1, halves time and frequency resolution, increases channels
        self.down1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)

        # 2nd encoder stage
        self.enc2 = MBConvBlock(64, 64, expansion=4, kernel_size=3)

        # dwonsample 2
        self.down2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)

        # 3rd encoder stage
        self.enc3 = MBConvBlock(128, 128, expansion=4, kernel_size=5)

        # downsample 3
        self.down3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)


        # bottleneck layer, stores compressed representation of spectrogram
        self.bottleneck = nn.Sequential(
            MBConvBlock(256, 256, expansion=4, kernel_size=5),
            MBConvBlock(256, 256, expansion=4, kernel_size=3)
        )

        # decoder

        # upsample 1, doubling the spatial size
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # 1st decoder stage, refines features after upsampling and reduces channel count
        self.dec1 = MBConvBlock(256, 128, expansion=4, kernel_size=5)

        # upsample 2
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # 2nd decoder stage 
        self.dec2 = MBConvBlock(128, 64, expansion=4, kernel_size=3)

        # upsample 3
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # 3rd decoder stage
        self.dec3 = MBConvBlock(64, 32, expansion=2, kernel_size=3)

        # final output layer
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    # forward pass
    def forward(self, x, mask):
        # concatenate masked spectrogram and mask along channel dimension
        inp = torch.cat([x, mask], dim=1)

        # initial feature extraction
        h = self.stem(inp)

        # encoder
        h = self.enc1(h)
        h = F.silu(self.down1(h))

        h = self.enc2(h)
        h = F.silu(self.down2(h))

        h = self.enc3(h)
        h = F.silu(self.down3(h))

        # bottleneck
        h = self.bottleneck(h)

        # decoder
        h = self.up1(h)
        h = self.dec1(h)

        h = self.up2(h)
        h = self.dec2(h)

        h = self.up3(h)
        h = self.dec3(h)

        # map features back to a single reconstructed spectrogram channel
        out = self.final_conv(h)

        # safety step:
        # resize back if the output spatial size differs from the input
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


        return out

    # configuration helpers
    def get_loss(self, lossfn):
        if self.gap_seconds in (0.5, 2.0):
            return self.shortgap_loss
        elif self.gap_seconds in (3.0, 5.0):
            return self.longgap_loss
        else:
            raise ValueError(
                f"Invalid gap_seconds={self.gap_seconds}. "
                f"Choose from [0.5, 2.0, 3.0, 5.0]."
            )

    def config_optimiser(self):
        self.optimiser = Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return self.optimiser

    def get_device(self):
        if self.device_name is not None:
            return torch.device(self.device_name)
        return next(self.parameters()).device

    # loss + metrics
    def compute_loss(self, pred, target, mask, lossfn=None):
        # masked l1
        if lossfn == "masked_l1":
            total_loss, gap_loss, context_loss = masked_l1_loss(pred, target, mask)
            loss_dict = {
                "loss": total_loss.item(),
                "gap_loss": gap_loss.item(),
                "context_loss": context_loss.item(),
            }
        # masked l1 gradient
        elif lossfn == "masked_l1_grad":
            total_loss, gap_loss, context_loss, grad_loss = masked_l1_grad_loss(pred, target, mask)
            loss_dict = {
                "loss": total_loss.item(),
                "gap_loss": gap_loss.item(),
                "context_loss": context_loss.item(),
                "grad_loss": grad_loss.item(),
            }

        # masked huber
        elif lossfn == "masked_huber":
            total_loss, gap_loss, context_loss = masked_huber_loss(pred, target, mask)
            loss_dict = {
                "loss": total_loss.item(),
                "gap_loss": gap_loss.item(),
                "context_loss": context_loss.item(),
            }

        # maskd multiresolution l1
        elif lossfn == "masked_multires_l1":
            total_loss, gap_loss, context_loss = masked_multires_l1_loss(pred, target, mask)
            loss_dict = {
                "loss": total_loss.item(),
                "gap_loss": gap_loss.item(),
                "context_loss": context_loss.item(),
            }

        else:
            raise ValueError(f"Unknown loss_name: {lossfn}")

        return total_loss, loss_dict

    # compute evaluation metrics
    def compute_metrics(self, pred, target, mask):
        return {
            "gap_mae": masked_mae(pred, target, mask).item(),
            "gap_rmse": masked_rmse(pred, target, mask).item(),
            "full_mae": full_mae(pred, target).item(),
            "full_rmse": full_rmse(pred, target).item(),
            "psnr": psnr(pred, target).item(),
        }
    
    # empty running dict for model training and logging
    def empty_dict(self):
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

    # train model for one full epoch
    def train_epoch(self, dataloader, optimiser=None, device=None, lossfn=None, grad_clip=None):
        self.train()

        # optmimiser
        optimiser = optimiser or self.optimiser
        if optimiser is None:
            raise ValueError("Optimiser is not set. Call configure_optimiser() first.")

        # device
        device = device or self.get_device()
        grad_clip = self.grad_clip if grad_clip is None else grad_clip

        running = self.empty_dict()
        num_batches = 0

        # loop through all batches in the training set
        for batch in dataloader:
            # move input tensors to device
            x = batch["x"].to(device)       # masked spectrogram
            y = batch["y"].to(device)       # clean target spectrogram
            mask = batch["mask"].to(device) # binary gap mask

            # reset old gradients before computing new ones
            optimiser.zero_grad()

            # forward pass - predict reocnstructed spectrogram
            pred = self(x, mask)

            # compute chosen loss 
            total_loss, loss_dict = self.compute_loss(pred, y, mask, lossfn=lossfn)

            total_loss.backward() # backward pass compute gradients

            # gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

            optimiser.step() # update model parameters

            # compute evaluation metrics for this batch
            eval_dict = self.compute_metrics(pred, y, mask)

            # accumulate loss metrics
            for key in loss_dict:
                if key in running:
                    running[key] += loss_dict[key]

            # accumulate eval metrics
            for key in eval_dict:
                running[key] += eval_dict[key]

            num_batches += 1

        # average all metrics across batches
        return {k: v / max(num_batches, 1) for k, v in running.items()}

    # evaluate model for one full epoch
    @torch.no_grad()
    def eval_epoch(self, dataloader, device=None, lossfn=None):
        # set model to eval mode
        self.eval()

        device = device or self.get_device()
        running = self._empty_running_dict()
        num_batches = 0

        # loop over validation data
        for batch in dataloader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)

            # forward pass 
            pred = self(x, mask)

            # compute training loss values for logging
            _, loss_dict = self.compute_loss(pred, y, mask, lossfn=lossfn)
            eval_dict = self.compute_metrics(pred, y, mask) # compute eval metrics

            # accumualte metrics
            for key, value in loss_dict.items():
                if key in running:
                    running[key] += value

            for key, value in eval_dict.items():
                if key in running:
                    running[key] += value

            num_batches += 1
        
        # avergae all metrics across batches
        return {k: v / max(num_batches, 1) for k, v in running.items()}

    # full training loop
    def fit(self, train_loader, val_loader, n_epochs, checkpoint_dir, history_csv_path,
             optimiser=None, lossfn=None, monitor="val_gap_rmse", mode="min", patience=5,
             min_delta=1e-4, save_best_after_epoch=10, grad_clip=None, device=None):
        
        device = device or self.get_device()
        optimiser = optimiser or self.optimiser
        if optimiser is None:
            optimiser = self.configure_optimiser()
        
        # history dictionary storing one list per metric
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

        # manager handles best model, last model and early stopping
        manager = ModelCheckpoint(
            checkpoint_dir=checkpoint_dir,
            monitor=monitor,
            mode=mode,
            patience=patience,
            min_delta=min_delta,
            save_best_after_epoch=save_best_after_epoch,
            verbose=True,
        )
        # main trianingn loop
        for epoch in tqdm(range(1, n_epochs + 1), desc="Training"):
            # train for one epoch
            train_metrics = self.train_epoch(
                dataloader=train_loader,
                optimiser=optimiser,
                device=device,
                lossfn=lossfn,
                grad_clip=grad_clip,
            )
            # validate for one epoch
            val_metrics = self.eval_epoch(
                dataloader=val_loader,
                device=device,
                lossfn=lossfn,
            )
            # compute train and validation metrics into one record
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

            # append values into history lists
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

            # checkpoint + early stopping step
            manager.step(
                epoch=epoch,
                metrics=epoch_record,
                model=self,
                optimiser=optimiser,
            )
            
            # stop training if patience is exhausted
            if manager.should_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # save full training history to csv
        save_history(history, history_csv_path)

        return {
            "history": history,
            "best_score": manager.best_score,
            "best_epoch": manager.best_epoch,
            "checkpoint_dir": checkpoint_dir,
        }

    # inferencing helpers
    @torch.no_grad()
    # inferencing step
    def inference(self, x, mask, device=None):
        # set to eval mode
        self.eval()
        device = device or self.get_device()

        # if single sample has no batch dimension, add one
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        
        # move to device
        x = x.to(device)
        mask = mask.to(device)

        # forward pass
        pred = self(x, mask)
        return pred

    @torch.no_grad()
    # mix prediction with context
    def reconstruct_full(self, x, mask, device=None):
        pred = self.inference(x, mask, device=device)
        x_device = x.to(pred.device)
        mask_device = mask.to(pred.device)

        if x_device.dim() == 3:
            x_device = x_device.unsqueeze(0)
        if mask_device.dim() == 3:
            mask_device = mask_device.unsqueeze(0)

        return x_device * (1.0 - mask_device) + pred * mask_device