from __future__ import annotations
# import packages

# standard library
import os
import math
import csv
import random
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchinfo import summary
from tqdm.auto import tqdm
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

    # diffusion losses
    diffusion_noise_mse_loss,
    diffusion_noise_l1_loss,
    diffusion_noise_huber_loss,
    masked_diffusion_noise_mse_loss,
    per_sample_mse_loss,
    per_sample_masked_mse_loss,
    min_snr,

    # latent losses
    latent_l1_loss,
    latent_l2_loss,

    # evaluation metrics
    masked_mae,
    masked_rmse,
    full_mae,
    full_rmse,
    psnr
)

__all__ = ["RetrievalBank", "RetrievalConditionedDiffusionUNet"]


# private utilities
def _require_external(name: str):
    obj = globals().get(name, None)
    if obj is None:
        raise NameError(
            f"Missing external utility `{name}`. Import it at the top of "
            "enhance_diffusion.py from your own training/loss utility file."
        )
    return obj


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def get_lr(optimiser: torch.optim.Optimizer) -> float:
    return optimiser.param_groups[0]["lr"]


def compute_gap_ratio(mask: torch.Tensor) -> torch.Tensor:
    return mask.float().mean(dim=(1, 2, 3), keepdim=True)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Sequence[int]) -> torch.Tensor:
    b = t.shape[0]
    out = a.gather(0, t)
    reshape_dims = (b,) + (1,) * (len(x_shape) - 1)
    return out.view(*reshape_dims)


def padding(x: torch.Tensor, multiple: int = 8) -> Tuple[torch.Tensor, Dict[str, int]]:
    _, _, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return x_pad, {"orig_h": h, "orig_w": w, "pad_h": pad_h, "pad_w": pad_w}


def unpadding(x: torch.Tensor, pad_info: Dict[str, int]) -> torch.Tensor:
    return x[..., : pad_info["orig_h"], : pad_info["orig_w"]]


@torch.no_grad()
def encode2latentmean(vae: nn.Module, x: torch.Tensor) -> torch.Tensor:
    mu, _ = vae.encode(x)
    return mu


@torch.no_grad()
def decode_from_latent(vae: nn.Module, z: torch.Tensor) -> torch.Tensor:
    return vae.decode(z)


def _device_type(device: torch.device | str) -> str:
    return torch.device(device).type


def _schedule_to_device(schedule: SimpleNamespace, device: torch.device | str) -> SimpleNamespace:
    out = {}
    for key, value in vars(schedule).items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return SimpleNamespace(**out)


# Retrieval bank
class RetrievalBank:
    DEFAULT_METADATA_KEYS = (
        "example_id",
        "audio_filename",
        "recording_idx",
        "clip_idx",
        "gap_seconds",
    )

    def __init__(
        self,
        device: torch.device | str = "cpu",
        window_size: int = 64,
        pooled_hw: Tuple[int, int] = (8, 8),
        metadata_keys: Sequence[str] = DEFAULT_METADATA_KEYS,
    ):
        self.device = torch.device(device)
        self.window_size = window_size
        self.pooled_hw = pooled_hw
        self.metadata_keys = tuple(metadata_keys)

        self.bank_embeds: Optional[torch.Tensor] = None
        self.bank_contexts: Optional[torch.Tensor] = None
        self.bank_targets: Optional[torch.Tensor] = None
        self.bank_masks: Optional[torch.Tensor] = None
        self.bank_meta: Optional[List[Dict[str, Any]]] = None

    @property
    def is_built(self) -> bool:
        return self.bank_embeds is not None and self.bank_targets is not None

    def __len__(self) -> int:
        if self.bank_embeds is None:
            return 0
        return int(self.bank_embeds.shape[0])

    def to(self, device: torch.device | str, tensors: str = "embeds_only") -> "RetrievalBank":
        """
        Move stored tensors.

        """
        device = torch.device("cpu" if tensors == "cpu" else device)
        self.device = device

        names = ["bank_embeds"] if tensors == "embeds_only" else [
            "bank_embeds",
            "bank_contexts",
            "bank_targets",
            "bank_masks",
        ]
        for name in names:
            value = getattr(self, name)
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        return self

    @torch.no_grad()
    def build(
        self,
        vae: nn.Module,
        dataloader: Iterable[Dict[str, Any]],
        device: torch.device | str,
        max_items: int = 4000,
        desc: str = "building retrieval bank",
        window_size: Optional[int] = None,
        pooled_hw: Optional[Tuple[int, int]] = None,
        pad_multiple: int = 8,
        store_device: torch.device | str = "cpu",
    ) -> "RetrievalBank":
        """
        build the retrieval bank from masked inputs and clean targets
        """
        window_size = window_size or self.window_size
        pooled_hw = pooled_hw or self.pooled_hw
        store_device = torch.device(store_device)
        device = torch.device(device)

        embeds: List[torch.Tensor] = []
        contexts: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        metas: List[Dict[str, Any]] = []

        vae.eval()
        n_total = 0

        for batch in tqdm(dataloader, desc=desc):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)

            x, _ = padding(x, multiple=pad_multiple)
            y, _ = padding(y, multiple=pad_multiple)
            m, _ = padding(m, multiple=pad_multiple)

            z_masked = encode2latentmean(vae, x)
            z_clean = encode2latentmean(vae, y)

            m_latent = F.interpolate(m, size=z_masked.shape[-2:], mode="nearest")
            z_context = z_masked * (1.0 - m_latent)

            emb = self.build_retrieval_emb(
                z_context=z_context,
                m_latent=m_latent,
                window_size=window_size,
                pooled_hw=pooled_hw,
            )

            embeds.append(emb.to(store_device))
            contexts.append(z_context.to(store_device))
            targets.append(z_clean.to(store_device))
            masks.append(m_latent.to(store_device))

            batch_size = x.shape[0]
            metas.extend(self.batch_query_meta_from_batch(batch, batch_size))

            n_total += batch_size
            if n_total >= max_items:
                break

        if len(embeds) == 0:
            raise RuntimeError("RetrievalBank.build() received no batches/items.")

        self.bank_embeds = torch.cat(embeds, dim=0)[:max_items].contiguous()
        self.bank_contexts = torch.cat(contexts, dim=0)[:max_items].contiguous()
        self.bank_targets = torch.cat(targets, dim=0)[:max_items].contiguous()
        self.bank_masks = torch.cat(masks, dim=0)[:max_items].contiguous()
        self.bank_meta = metas[:max_items]
        self.device = store_device

        print(f"built retrieval bank with {len(self)} items")
        return self

    def save(self, path: str | Path) -> None:
        """save retrieval bank tensors and metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "window_size": self.window_size,
                "pooled_hw": self.pooled_hw,
                "metadata_keys": self.metadata_keys,
                "bank_embeds": self.bank_embeds,
                "bank_contexts": self.bank_contexts,
                "bank_targets": self.bank_targets,
                "bank_masks": self.bank_masks,
                "bank_meta": self.bank_meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> "RetrievalBank":
        """Load a previously saved retrieval bank."""
        ckpt = torch.load(path, map_location=device)
        bank = cls(
            device=device,
            window_size=ckpt.get("window_size", 64),
            pooled_hw=tuple(ckpt.get("pooled_hw", (8, 8))),
            metadata_keys=tuple(ckpt.get("metadata_keys", cls.DEFAULT_METADATA_KEYS)),
        )
        bank.bank_embeds = ckpt["bank_embeds"]
        bank.bank_contexts = ckpt["bank_contexts"]
        bank.bank_targets = ckpt["bank_targets"]
        bank.bank_masks = ckpt["bank_masks"]
        bank.bank_meta = ckpt.get("bank_meta", None)
        return bank.to(device, tensors="all")

    @torch.no_grad()
    def query(
        self,
        query_context_latent: torch.Tensor,
        query_mask_latent: torch.Tensor,
        top_k: int = 5,
        query_meta: Optional[List[Dict[str, Any]]] = None,
        window_size: Optional[int] = None,
        pooled_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Query top-k similar retrieval examples.

        Returns:
            retrieved_top1_target: [B, C, H, W]
            sim_top1:               [B, 1]
            top_scores:             [B, K] on CPU
            top_idx:                [B, K] on CPU
        """
        if not self.is_built:
            raise RuntimeError("Build or load the retrieval bank before calling query().")

        window_size = window_size or self.window_size
        pooled_hw = pooled_hw or self.pooled_hw

        q = self.build_retrieval_emb(
            z_context=query_context_latent,
            m_latent=query_mask_latent,
            window_size=window_size,
            pooled_hw=pooled_hw,
        )

        bank_embeds = self.bank_embeds
        assert bank_embeds is not None

        sims = torch.matmul(q.cpu(), bank_embeds.cpu().T)  # [B, N]
        bsz, n_items = sims.shape
        overfetch_k = min(max(top_k * 8, top_k + 10), n_items)

        raw_scores, raw_idx = torch.topk(sims, k=overfetch_k, dim=1)
        final_scores: List[torch.Tensor] = []
        final_idx: List[torch.Tensor] = []

        for b in range(bsz):
            kept_scores: List[float] = []
            kept_indices: List[int] = []
            q_meta = query_meta[b] if query_meta is not None else None

            for score, idx in zip(raw_scores[b].tolist(), raw_idx[b].tolist()):
                bank_meta = self.bank_meta[idx] if self.bank_meta is not None else None

                if q_meta is not None and bank_meta is not None:
                    if self.same_source_clip(q_meta, bank_meta):
                        continue

                kept_scores.append(float(score))
                kept_indices.append(int(idx))

                if len(kept_indices) >= top_k:
                    break

            if len(kept_indices) == 0:
                kept_scores.append(float(raw_scores[b, 0].item()))
                kept_indices.append(int(raw_idx[b, 0].item()))

            while len(kept_indices) < top_k:
                kept_scores.append(0.0)
                kept_indices.append(kept_indices[-1])

            final_scores.append(torch.tensor(kept_scores, dtype=torch.float32))
            final_idx.append(torch.tensor(kept_indices, dtype=torch.long))

        top_scores = torch.stack(final_scores, dim=0)
        top_idx = torch.stack(final_idx, dim=0)

        assert self.bank_targets is not None
        retrieved_top1_target = self.bank_targets[top_idx[:, 0]].to(query_context_latent.device)
        sim_top1 = top_scores[:, 0].to(query_context_latent.device).unsqueeze(1)

        return {
            "retrieved_top1_target": retrieved_top1_target,
            "sim_top1": sim_top1,
            "top_scores": top_scores,
            "top_idx": top_idx,
        }

    @torch.no_grad()
    def diversify_topk_indices(
        self,
        top_idx: torch.Tensor,
        top_scores: torch.Tensor,
        max_keep: int = 3,
        similarity_threshold: float = 0.92,
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """Greedy diversity filter over the retrieved candidates."""
        if self.bank_embeds is None:
            raise RuntimeError("bank_embeds is missing. Build or load the bank first.")

        kept_idx_list: List[List[int]] = []
        kept_score_list: List[List[float]] = []
        bank_embeds_cpu = self.bank_embeds.cpu()

        for b in range(top_idx.shape[0]):
            cand_idx = top_idx[b].tolist()
            cand_scores = top_scores[b].tolist()

            selected_idx: List[int] = []
            selected_scores: List[float] = []
            selected_embeds: List[torch.Tensor] = []

            for idx, sc in zip(cand_idx, cand_scores):
                emb = bank_embeds_cpu[idx]

                keep = True
                for prev_emb in selected_embeds:
                    sim = torch.dot(emb, prev_emb).item()
                    if sim >= similarity_threshold:
                        keep = False
                        break

                if keep:
                    selected_idx.append(int(idx))
                    selected_scores.append(float(sc))
                    selected_embeds.append(emb)

                if len(selected_idx) >= max_keep:
                    break

            if len(selected_idx) == 0:
                selected_idx.append(int(cand_idx[0]))
                selected_scores.append(float(cand_scores[0]))

            kept_idx_list.append(selected_idx)
            kept_score_list.append(selected_scores)

        return kept_idx_list, kept_score_list

    @torch.no_grad()
    def build_topk_candidates(
        self,
        z_context: torch.Tensor,
        m_latent: torch.Tensor,
        device: torch.device | str,
        top_k: int = 5,
        max_keep: int = 3,
        diversity_threshold: float = 0.92,
        query_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        retrieve, diversity-filter, and pack variable-length candidates into tensors.
        """
        if self.bank_targets is None:
            raise RuntimeError("bank_targets is missing. Build or load the bank first.")

        device = torch.device(device)
        retrieval_out = self.query(
            query_context_latent=z_context,
            query_mask_latent=m_latent,
            top_k=top_k,
            query_meta=query_meta,
        )

        top_idx = retrieval_out["top_idx"].cpu()
        top_scores = retrieval_out["top_scores"].cpu()

        kept_idx_list, kept_score_list = self.diversify_topk_indices(
            top_idx=top_idx,
            top_scores=top_scores,
            max_keep=max_keep,
            similarity_threshold=diversity_threshold,
        )

        bsz = len(kept_idx_list)
        max_candidates = max(len(x) for x in kept_idx_list)

        candidate_targets: List[torch.Tensor] = []
        candidate_scores: List[torch.Tensor] = []
        candidate_valid_mask: List[torch.Tensor] = []
        candidate_indices: List[List[int]] = []

        for b in range(bsz):
            idxs = kept_idx_list[b]
            scs = kept_score_list[b]

            tgts = self.bank_targets[idxs].to(device)
            scs_t = torch.tensor(scs, dtype=torch.float32, device=device)
            valid = torch.ones(len(idxs), dtype=torch.float32, device=device)

            if len(idxs) < max_candidates:
                pad_n = max_candidates - len(idxs)
                tgts = torch.cat(
                    [tgts, torch.zeros((pad_n, *tgts.shape[1:]), device=device)],
                    dim=0,
                )
                scs_t = torch.cat([scs_t, torch.zeros(pad_n, device=device)], dim=0)
                valid = torch.cat([valid, torch.zeros(pad_n, device=device)], dim=0)
                idxs = idxs + [-1] * pad_n

            candidate_targets.append(tgts.unsqueeze(0))
            candidate_scores.append(scs_t.unsqueeze(0))
            candidate_valid_mask.append(valid.unsqueeze(0))
            candidate_indices.append(idxs)

        return {
            "candidate_targets": torch.cat(candidate_targets, dim=0),
            "candidate_scores": torch.cat(candidate_scores, dim=0),
            "candidate_valid_mask": torch.cat(candidate_valid_mask, dim=0),
            "candidate_indices": candidate_indices,
            "raw_top_scores": retrieval_out["top_scores"],
            "raw_top_idx": retrieval_out["top_idx"],
        }

    @staticmethod
    def find_gap_from_mask(mask_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, _, W = mask_latent.shape
        gap_1d = (mask_latent.mean(dim=2) > 0.5).squeeze(1)

        start_idx = torch.zeros(B, dtype=torch.long, device=mask_latent.device)
        end_idx = torch.full((B,), W, dtype=torch.long, device=mask_latent.device)

        for b in range(B):
            idx = torch.where(gap_1d[b])[0]
            if len(idx) > 0:
                start_idx[b] = idx[0]
                end_idx[b] = idx[-1] + 1

        return start_idx, end_idx

    @classmethod
    def extract_gap_window(
        cls,
        z_context: torch.Tensor,
        m_latent: torch.Tensor,
        window_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, _, W = z_context.shape
        gap_start, gap_end = cls.find_gap_from_mask(m_latent)
        known_mask = 1.0 - m_latent

        z_left_list: List[torch.Tensor] = []
        z_right_list: List[torch.Tensor] = []
        m_left_list: List[torch.Tensor] = []
        m_right_list: List[torch.Tensor] = []

        for b in range(B):
            s = int(gap_start[b].item())
            e = int(gap_end[b].item())

            left_start = max(0, s - window_size)
            left_end = s
            right_start = e
            right_end = min(W, e + window_size)

            z_left = z_context[b : b + 1, :, :, left_start:left_end]
            z_right = z_context[b : b + 1, :, :, right_start:right_end]
            m_left = known_mask[b : b + 1, :, :, left_start:left_end]
            m_right = known_mask[b : b + 1, :, :, right_start:right_end]

            if z_left.shape[-1] < window_size:
                pad = window_size - z_left.shape[-1]
                z_left = F.pad(z_left, (pad, 0, 0, 0), value=0.0)
                m_left = F.pad(m_left, (pad, 0, 0, 0), value=0.0)

            if z_right.shape[-1] < window_size:
                pad = window_size - z_right.shape[-1]
                z_right = F.pad(z_right, (0, pad, 0, 0), value=0.0)
                m_right = F.pad(m_right, (0, pad, 0, 0), value=0.0)

            z_left_list.append(z_left)
            z_right_list.append(z_right)
            m_left_list.append(m_left)
            m_right_list.append(m_right)

        return (
            torch.cat(z_left_list, dim=0),
            torch.cat(z_right_list, dim=0),
            torch.cat(m_left_list, dim=0),
            torch.cat(m_right_list, dim=0),
        )

    @classmethod
    def build_retrieval_emb(
        cls,
        z_context: torch.Tensor,
        m_latent: torch.Tensor,
        window_size: int = 64,
        pooled_hw: Tuple[int, int] = (8, 8),
    ) -> torch.Tensor:
        """
        build a normalised retrieval embedding from local latent context.
        """
        z_left, z_right, m_left, m_right = cls.extract_gap_window(
            z_context=z_context,
            m_latent=m_latent,
            window_size=window_size,
        )

        z_left = z_left * m_left
        z_right = z_right * m_right

        left_pool = F.adaptive_avg_pool2d(z_left, pooled_hw)
        right_pool = F.adaptive_avg_pool2d(z_right, pooled_hw)

        emb = torch.cat([left_pool.flatten(1), right_pool.flatten(1)], dim=1)
        return F.normalize(emb, dim=1)

    @staticmethod
    def same_source_clip(meta_a: Optional[Dict[str, Any]], meta_b: Optional[Dict[str, Any]]) -> bool:
        """conservative duplicate check using example_id or source/clip identifiers."""
        if meta_a is None or meta_b is None:
            return False

        if "example_id" in meta_a and "example_id" in meta_b:
            if meta_a["example_id"] == meta_b["example_id"]:
                return True

        keys = ["audio_filename", "recording_idx", "clip_idx"]
        if all(k in meta_a for k in keys) and all(k in meta_b for k in keys):
            return (
                meta_a["audio_filename"] == meta_b["audio_filename"]
                and meta_a["recording_idx"] == meta_b["recording_idx"]
                and meta_a["clip_idx"] == meta_b["clip_idx"]
            )

        return False

    def batch_query_meta_from_batch(self, batch: Dict[str, Any], batch_size: int) -> List[Dict[str, Any]]:
        metas: List[Dict[str, Any]] = []
        for i in range(batch_size):
            meta: Dict[str, Any] = {}
            for key in self.metadata_keys:
                if key in batch:
                    value = batch[key]
                    if torch.is_tensor(value):
                        try:
                            meta[key] = value[i].item()
                        except Exception:
                            meta[key] = value[i]
                    else:
                        meta[key] = value[i]
            metas.append(meta)
        return metas


# Retrieval-Conditioned Diffusion UNet
class RetrievalConditionedDiffusionUNet(nn.Module):
    """
    retrieval-conditioned latent diffusion U-Net
    """
    # inetenal architecture blocks
    class _SinusoidalTimeEmb(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim

        def forward(self, t: torch.Tensor) -> torch.Tensor:
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

    class _TimeEmbMLP(nn.Module):
        def __init__(self, time_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(time_dim, time_dim * 4),
                nn.SiLU(),
                nn.Linear(time_dim * 4, time_dim),
            )

        def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
            return self.net(t_emb)

    class _SEBlock(nn.Module):
        def __init__(self, channels: int, reduction: int = 4):
            super().__init__()
            hidden = max(channels // reduction, 8)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
            self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            scale = self.pool(x)
            scale = F.silu(self.fc1(scale))
            scale = torch.sigmoid(self.fc2(scale))
            return x * scale

    class _DownSample(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    class _UpSample(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            return self.conv(x)

    class _DiffusionResBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, use_se: bool = True):
            super().__init__()
            self.norm1 = make_norm(in_channels)
            self.act1 = nn.SiLU()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

            self.time_proj = nn.Linear(time_emb_dim, out_channels * 2)

            self.norm2 = make_norm(out_channels)
            self.act2 = nn.SiLU()
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

            self.se = RetrievalConditionedDiffusionUNet._SEBlock(out_channels) if use_se else nn.Identity()
            self.skip = (
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
                if in_channels != out_channels
                else nn.Identity()
            )

        def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
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

    class _DilatedResBlock(nn.Module):
        def __init__(self, channels: int, dilation: int = 2):
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            h = self.norm1(x)
            h = self.act1(h)
            h = self.conv1(h)
            h = self.norm2(h)
            h = self.act2(h)
            h = self.conv2(h)
            return h + residual

    class _SelfAttention2D(nn.Module):
        def __init__(self, channels: int, num_heads: int = 4):
            super().__init__()
            assert channels % num_heads == 0
            self.channels = channels
            self.num_heads = num_heads
            self.head_dim = channels // num_heads

            self.norm = make_norm(channels)
            self.to_q = nn.Conv2d(channels, channels, kernel_size=1)
            self.to_k = nn.Conv2d(channels, channels, kernel_size=1)
            self.to_v = nn.Conv2d(channels, channels, kernel_size=1)
            self.proj = nn.Conv2d(channels, channels, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, c, h, w = x.shape
            residual = x
            x = self.norm(x)

            q = self.to_q(x).view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
            k = self.to_k(x).view(b, self.num_heads, self.head_dim, h * w)
            v = self.to_v(x).view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)

            attn = torch.matmul(q, k) / math.sqrt(self.head_dim)
            attn = torch.softmax(attn, dim=-1)

            out = torch.matmul(attn, v)
            out = out.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
            out = self.proj(out)
            return out + residual

    class _CrossAttentionBlock(nn.Module):
        def __init__(self, channels: int, cond_dim: int, num_heads: int = 4):
            super().__init__()
            assert channels % num_heads == 0
            self.channels = channels
            self.cond_dim = cond_dim
            self.num_heads = num_heads
            self.head_dim = channels // num_heads

            self.norm = make_norm(channels)
            self.to_q = nn.Conv2d(channels, channels, kernel_size=1)
            self.to_k = nn.Linear(cond_dim, channels)
            self.to_v = nn.Linear(cond_dim, channels)
            self.proj = nn.Conv2d(channels, channels, kernel_size=1)

        def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
            b, c, h, w = x.shape
            residual = x
            x_norm = self.norm(x)

            q = self.to_q(x_norm).view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
            k = self.to_k(cond_tokens).view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
            v = self.to_v(cond_tokens).view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            attn = torch.matmul(q, k) / math.sqrt(self.head_dim)
            attn = torch.softmax(attn, dim=-1)

            out = torch.matmul(attn, v)
            out = out.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
            out = self.proj(out)
            return out + residual

    class _RetrievalEncoder(nn.Module):
        def __init__(self, latent_channels: int, cond_dim: int = 256, base_channels: int = 64):
            super().__init__()
            self.in_conv = nn.Conv2d(latent_channels, base_channels, kernel_size=3, padding=1)
            self.block1 = nn.Sequential(
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                make_norm(base_channels),
                nn.SiLU(),
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                make_norm(base_channels),
                nn.SiLU(),
            )
            self.down1 = RetrievalConditionedDiffusionUNet._DownSample(base_channels)
            self.block2 = nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
                make_norm(base_channels * 2),
                nn.SiLU(),
                nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
                make_norm(base_channels * 2),
                nn.SiLU(),
            )
            self.proj = nn.Conv2d(base_channels * 2, cond_dim, kernel_size=1)

        def forward(self, z_retrieved: torch.Tensor) -> torch.Tensor:
            x = self.in_conv(z_retrieved)
            x = self.block1(x)
            x = self.down1(x)
            x = self.block2(x)
            x = self.proj(x)
            b, d, h, w = x.shape
            return x.flatten(2).transpose(1, 2).contiguous()  # [B, N, D]

    class _LearnedFusionModule(nn.Module):
        def __init__(self, latent_channels: int = 8, hidden_dim: int = 128):
            super().__init__()
            in_dim = latent_channels * 3 + 2
            self.score_net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

        @staticmethod
        def global_pool(x: torch.Tensor) -> torch.Tensor:
            return F.adaptive_avg_pool2d(x, output_size=1).flatten(1)

        @staticmethod
        def gap_pool(x: torch.Tensor, m_latent: torch.Tensor) -> torch.Tensor:
            gap_mask = m_latent.expand(-1, x.shape[1], -1, -1).to(x.dtype)
            gap_sum = (x * gap_mask).sum(dim=(-2, -1))
            gap_count = gap_mask.sum(dim=(-2, -1)).clamp(min=1.0)
            return gap_sum / gap_count

        def forward(
            self,
            candidate_targets: torch.Tensor,
            candidate_scores: torch.Tensor,
            candidate_valid_mask: torch.Tensor,
            z_context: torch.Tensor,
            m_latent: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            bsz, k, _, _, _ = candidate_targets.shape
            context_global_vec = self.global_pool(z_context)
            gap_ratio = compute_gap_ratio(m_latent).view(bsz, 1).to(z_context.dtype)

            logits = []
            for i in range(k):
                cand = candidate_targets[:, i]
                score_i = candidate_scores[:, i : i + 1].to(z_context.dtype)
                cand_gap_vec = self.gap_pool(cand, m_latent)
                cand_global_vec = self.global_pool(cand)
                feat = torch.cat(
                    [context_global_vec, cand_gap_vec, cand_global_vec, score_i, gap_ratio],
                    dim=1,
                )
                logits.append(self.score_net(feat))

            logits = torch.cat(logits, dim=1).float()
            logits = torch.where(
                candidate_valid_mask > 0,
                logits,
                torch.full_like(logits, -1e4),
            )
            weights = torch.softmax(logits, dim=1)
            fused = (weights[:, :, None, None, None].to(candidate_targets.dtype) * candidate_targets).sum(dim=1)

            return {"fused_target": fused, "weights": weights, "logits": logits}

    # model construction
    def __init__(
        self,
        latent_channels: int = 8,
        base_channels: int = 128,
        time_dim: int = 256,
        cond_dim: int = 256,
        retrieval_base_channels: int = 64,
        num_retrieval_cross_attn: int = 2,
        fusion_mode: str = "selective_mix",
    ):
        super().__init__()
        if fusion_mode not in {"selective_mix", "learned_fusion"}:
            raise ValueError("fusion_mode must be 'selective_mix' or 'learned_fusion'")

        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        self.num_retrieval_cross_attn = num_retrieval_cross_attn
        self.fusion_mode = fusion_mode

        self.retrieval_encoder = self._RetrievalEncoder(
            latent_channels=latent_channels,
            cond_dim=cond_dim,
            base_channels=retrieval_base_channels,
        )

        self.fusion_module = (
            self._LearnedFusionModule(latent_channels=latent_channels, hidden_dim=128)
            if fusion_mode == "learned_fusion"
            else None
        )

        self.null_cond_token = nn.Parameter(torch.zeros(1, 1, cond_dim))

        in_channels = latent_channels + latent_channels + latent_channels + 1
        out_channels = latent_channels

        self.time_embed = self._SinusoidalTimeEmb(time_dim)
        self.time_mlp = self._TimeEmbMLP(time_dim)

        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down1_block1 = self._DiffusionResBlock(base_channels, base_channels, time_emb_dim=time_dim, use_se=True)
        self.down1_block2 = self._DiffusionResBlock(base_channels, base_channels, time_emb_dim=time_dim, use_se=True)
        self.down1 = self._DownSample(base_channels)

        self.down2_block1 = self._DiffusionResBlock(base_channels, base_channels * 2, time_emb_dim=time_dim, use_se=True)
        self.down2_block2 = self._DiffusionResBlock(base_channels * 2, base_channels * 2, time_emb_dim=time_dim, use_se=True)
        self.down2 = self._DownSample(base_channels * 2)

        self.down3_block1 = self._DiffusionResBlock(base_channels * 2, base_channels * 4, time_emb_dim=time_dim, use_se=True)
        self.down3_block2 = self._DiffusionResBlock(base_channels * 4, base_channels * 4, time_emb_dim=time_dim, use_se=True)
        self.down3 = self._DownSample(base_channels * 4)

        self.mid_block1 = self._DiffusionResBlock(base_channels * 4, base_channels * 4, time_emb_dim=time_dim, use_se=True)
        self.mid_dilate1 = self._DilatedResBlock(base_channels * 4, dilation=2)
        self.mid_self_attn = self._SelfAttention2D(base_channels * 4, num_heads=4)
        self.mid_retrieval_attn = self._CrossAttentionBlock(base_channels * 4, cond_dim=cond_dim, num_heads=4)
        self.mid_dilate2 = self._DilatedResBlock(base_channels * 4, dilation=4)
        self.mid_block2 = self._DiffusionResBlock(base_channels * 4, base_channels * 4, time_emb_dim=time_dim, use_se=True)

        self.up3 = self._UpSample(base_channels * 4)
        self.up3_block1 = self._DiffusionResBlock(base_channels * 8, base_channels * 4, time_emb_dim=time_dim, use_se=True)
        self.up3_retrieval_attn = (
            self._CrossAttentionBlock(base_channels * 4, cond_dim=cond_dim, num_heads=4)
            if num_retrieval_cross_attn >= 2
            else None
        )
        self.up3_block2 = self._DiffusionResBlock(base_channels * 4, base_channels * 2, time_emb_dim=time_dim, use_se=True)

        self.up2 = self._UpSample(base_channels * 2)
        self.up2_block1 = self._DiffusionResBlock(base_channels * 4, base_channels * 2, time_emb_dim=time_dim, use_se=True)
        self.up2_block2 = self._DiffusionResBlock(base_channels * 2, base_channels, time_emb_dim=time_dim, use_se=True)

        self.up1 = self._UpSample(base_channels)
        self.up1_block1 = self._DiffusionResBlock(base_channels * 2, base_channels, time_emb_dim=time_dim, use_se=True)
        self.up1_block2 = self._DiffusionResBlock(base_channels, base_channels, time_emb_dim=time_dim, use_se=True)

        self.out_norm = make_norm(base_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

        self._compiled = False
        self._compile_state: Dict[str, Any] = {}
        self._fit_config: Dict[str, Any] = self.default_fit_config()

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        h = self.mid_self_attn(h)

        if cond_tokens is not None:
            h = self.mid_retrieval_attn(h, cond_tokens)

        h = self.mid_dilate2(h)
        h = self.mid_block2(h, t_emb)

        h = self.up3(h)
        h = self.match_spatial(h, d3)
        h = torch.cat([h, d3], dim=1)
        h = self.up3_block1(h, t_emb)

        if self.up3_retrieval_attn is not None and cond_tokens is not None:
            h = self.up3_retrieval_attn(h, cond_tokens)

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
        return self.out_conv(h)

    @staticmethod
    def match_spatial(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="nearest")
        return x

    # diffusion schedule and diffusion helpers
    @staticmethod
    def cosine_schedule(
        num_steps: int,
        s: float = 0.008,
        max_beta: float = 0.999,
        device: torch.device | str = "cpu",
    ) -> SimpleNamespace:
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

        alpha_bars_prev = torch.cat([torch.tensor([1.0], device=device), alpha_bars[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        return SimpleNamespace(
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
            sqrt_alpha_bars=sqrt_alpha_bars,
            sqrt_one_minus_alpha_bars=sqrt_one_minus_alpha_bars,
            sqrt_recip_alphas=sqrt_recip_alphas,
            posterior_variance=posterior_variance,
        )

    @staticmethod
    def q_sample(z_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, schedule: SimpleNamespace) -> torch.Tensor:
        sqrt_alpha_bar_t = extract(schedule.sqrt_alpha_bars, t, z_0.shape)
        sqrt_one_minus_alpha_bar_t = extract(schedule.sqrt_one_minus_alpha_bars, t, z_0.shape)
        return sqrt_alpha_bar_t * z_0 + sqrt_one_minus_alpha_bar_t * noise

    @staticmethod
    def predict_x0(z_t: torch.Tensor, eps_hat: torch.Tensor, t: torch.Tensor, schedule: SimpleNamespace) -> torch.Tensor:
        sqrt_alpha_bar_t = extract(schedule.sqrt_alpha_bars, t, z_t.shape)
        sqrt_one_minus_alpha_bar_t = extract(schedule.sqrt_one_minus_alpha_bars, t, z_t.shape)
        return (z_t - sqrt_one_minus_alpha_bar_t * eps_hat) / (sqrt_alpha_bar_t + 1e-8)

    @classmethod
    def make_known_latent(
        cls,
        z_known_0: torch.Tensor,
        t: torch.Tensor,
        schedule: SimpleNamespace,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(z_known_0)
        return cls.q_sample(z_known_0, t, noise, schedule)

    @staticmethod
    def preserve_known_region(z_sample: torch.Tensor, z_known: torch.Tensor, mask_latent: torch.Tensor) -> torch.Tensor:
        return mask_latent * z_sample + (1.0 - mask_latent) * z_known

    # fusion and conditioning
    @staticmethod
    @torch.no_grad()
    def heuristic_selective_mix(
        candidate_targets: torch.Tensor,
        candidate_scores: torch.Tensor,
        candidate_valid_mask: Optional[torch.Tensor] = None,
        temperature: float = 0.08,
        sharpen_power: float = 2.0,
        min_keep_weight: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        scores = candidate_scores.float()

        if candidate_valid_mask is not None:
            scores = torch.where(
                candidate_valid_mask > 0,
                scores,
                torch.full_like(scores, -1e4),
            )

        weights = torch.softmax(scores / temperature, dim=1)
        weights = weights ** sharpen_power
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

        if min_keep_weight is not None and min_keep_weight > 0:
            keep_mask = weights >= min_keep_weight
            if candidate_valid_mask is not None:
                keep_mask = keep_mask & (candidate_valid_mask > 0)

            weights = torch.where(keep_mask, weights, torch.zeros_like(weights))

            row_sum = weights.sum(dim=1, keepdim=True)
            zero_rows = row_sum.squeeze(1) <= 1e-8

            if zero_rows.any():
                fallback_idx = torch.argmax(scores[zero_rows], dim=1)
                fallback = torch.zeros_like(weights[zero_rows])
                fallback.scatter_(1, fallback_idx.unsqueeze(1), 1.0)
                weights[zero_rows] = fallback

            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

        fused = (weights[:, :, None, None, None].to(candidate_targets.dtype) * candidate_targets).sum(dim=1)
        return {"fused_target": fused, "weights": weights}

    @staticmethod
    @torch.no_grad()
    def sample_retrieval_strength(
        batch_size: int,
        device: torch.device | str,
        p_none: float = 0.20,
        p_weak: float = 0.40,
        p_mid: float = 0.20,
        p_strong: float = 0.20,
        weak_range: Tuple[float, float] = (0.30, 0.50),
        mid_range: Tuple[float, float] = (0.55, 0.75),
        strong_range: Tuple[float, float] = (0.85, 1.00),
    ) -> torch.Tensor:
        device = torch.device(device)
        u = torch.rand(batch_size, device=device)
        r = torch.zeros(batch_size, device=device)

        weak_mask = (u >= p_none) & (u < p_none + p_weak)
        mid_mask = (u >= p_none + p_weak) & (u < p_none + p_weak + p_mid)
        strong_mask = u >= (p_none + p_weak + p_mid)

        if weak_mask.any():
            r[weak_mask] = weak_range[0] + (weak_range[1] - weak_range[0]) * torch.rand(weak_mask.sum(), device=device)
        if mid_mask.any():
            r[mid_mask] = mid_range[0] + (mid_range[1] - mid_range[0]) * torch.rand(mid_mask.sum(), device=device)
        if strong_mask.any():
            r[strong_mask] = strong_range[0] + (strong_range[1] - strong_range[0]) * torch.rand(strong_mask.sum(), device=device)

        return r.view(batch_size, 1, 1, 1)

    def build_retrieval_condition(
        self,
        candidate_targets: torch.Tensor,
        candidate_scores: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
        z_context: torch.Tensor,
        m_latent: torch.Tensor,
        retrieval_strength: torch.Tensor,
        selective_temperature: float = 0.08,
        selective_sharpen_power: float = 2.0,
        selective_min_keep_weight: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        if self.fusion_mode == "learned_fusion":
            assert self.fusion_module is not None
            fusion_out = self.fusion_module(
                candidate_targets=candidate_targets,
                candidate_scores=candidate_scores,
                candidate_valid_mask=candidate_valid_mask,
                z_context=z_context,
                m_latent=m_latent,
            )
            fused_target = fusion_out["fused_target"]
            fusion_weights = fusion_out["weights"]
            fusion_logits = fusion_out["logits"]
        else:
            fusion_out = self.heuristic_selective_mix(
                candidate_targets=candidate_targets,
                candidate_scores=candidate_scores,
                candidate_valid_mask=candidate_valid_mask,
                temperature=selective_temperature,
                sharpen_power=selective_sharpen_power,
                min_keep_weight=selective_min_keep_weight,
            )
            fused_target = fusion_out["fused_target"]
            fusion_weights = fusion_out["weights"]
            fusion_logits = candidate_scores

        scaled_target = fused_target * retrieval_strength
        cond_tokens = self.retrieval_encoder(scaled_target)

        strength = retrieval_strength.view(-1, 1, 1).to(cond_tokens.dtype)
        null_tokens = self.null_cond_token.to(cond_tokens.dtype).expand(
            cond_tokens.shape[0],
            cond_tokens.shape[1],
            -1,
        )
        cond_tokens = strength * cond_tokens + (1.0 - strength) * null_tokens

        return {
            "cond_tokens": cond_tokens,
            "fused_target": fused_target,
            "fusion_weights": fusion_weights,
            "fusion_logits": fusion_logits,
            "retrieval_strength": retrieval_strength,
        }

    # loss helpers
    @staticmethod
    def compute_noise_loss(
        pred_noise: torch.Tensor,
        true_noise: torch.Tensor,
        lossfn: str = "masked_mse",
        delta: float = 1.0,
        mask_latent: Optional[torch.Tensor] = None,
        masked_weight: float = 3.0,
        schedule: Optional[SimpleNamespace] = None,
        t: Optional[torch.Tensor] = None,
        use_min_snr: bool = True,
        min_snr_gamma: float = 5.0,
    ) -> torch.Tensor:
        if lossfn == "mse":
            if use_min_snr:
                per_sample = _require_external("per_sample_mse_loss")(pred_noise, true_noise)
                weight = _require_external("min_snr")(schedule, t, gamma=min_snr_gamma)
                return (per_sample * weight).mean()
            return _require_external("diffusion_noise_mse_loss")(pred_noise, true_noise)

        if lossfn == "masked_mse":
            if mask_latent is None:
                raise ValueError("mask_latent must be provided for masked_mse")

            if use_min_snr:
                per_sample = _require_external("per_sample_masked_mse_loss")(
                    pred=pred_noise,
                    target=true_noise,
                    mask_latent=mask_latent,
                    masked_weight=masked_weight,
                )
                weight = _require_external("min_snr")(schedule, t, gamma=min_snr_gamma)
                return (per_sample * weight).mean()

            return _require_external("masked_diffusion_noise_mse_loss")(
                pred_noise=pred_noise,
                true_noise=true_noise,
                mask_latent=mask_latent,
                masked_weight=masked_weight,
            )

        if lossfn == "l1":
            return _require_external("diffusion_noise_l1_loss")(pred_noise, true_noise)

        if lossfn == "huber":
            return _require_external("diffusion_noise_huber_loss")(pred_noise, true_noise, delta=delta)

        raise ValueError(f"Unsupported lossfn: {lossfn}")

    @staticmethod
    def compute_latent_loss(
        pred_latent: torch.Tensor,
        target_latent: torch.Tensor,
        mask_latent: torch.Tensor,
        latent_loss: str = "l1",
        context_weight: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if latent_loss == "l1":
            return _require_external("masked_latent_l1_loss")(
                pred_latent=pred_latent,
                target_latent=target_latent,
                mask_latent=mask_latent,
                context_weight=context_weight,
            )
        if latent_loss == "l2":
            return _require_external("masked_latent_l2_loss")(
                pred_latent=pred_latent,
                target_latent=target_latent,
                mask_latent=mask_latent,
                context_weight=context_weight,
            )
        raise ValueError(f"Unsupported latent loss: {latent_loss}")

    # training interface
    @staticmethod
    def default_fit_config() -> Dict[str, Any]:
        return {
            "noise_loss": "masked_mse",
            "latent_loss": "l1",
            "latent_loss_weight": 0.0,
            "delta": 1.0,
            "masked_weight": 3.0,
            "use_min_snr": True,
            "min_snr_gamma": 5.0,
            "use_self_condition": True,
            "p_selfcond": 0.5,
            "cond_dropout_prob": 0.10,
            "top_k": 5,
            "max_keep": 3,
            "diversity_threshold": 0.92,
            "selective_temperature": 0.08,
            "selective_sharpen_power": 2.0,
            "selective_min_keep_weight": 0.05,
            "retrieval_strength_cfg": None,
            "grad_clip": 1.0,
            "use_amp": True,
            "pad_multiple": 8,
            "latent_context_weight": 0.1,
        }

    def compile(
        self,
        *,
        vae: nn.Module,
        train_retrieval_bank: RetrievalBank,
        val_retrieval_bank: Optional[RetrievalBank] = None,
        schedule: Optional[SimpleNamespace] = None,
        num_steps: int = 1000,
        optimiser: Optional[torch.optim.Optimizer] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1.25e-5,
        weight_decay: float = 1e-4,
        device: Optional[torch.device | str] = None,
        freeze_vae: bool = True,
        seed: Optional[int] = 42,
        **fit_config: Any,
    ) -> "RetrievalConditionedDiffusionUNet":
        """
        Attach training objects to the model.

        This is intentionally similar to Keras-style compile(): it stores the VAE,
        retrieval banks, schedule, optimiser, and default training config.
        """
        if seed is not None:
            set_seed(seed)

        device = torch.device(device) if device is not None else next(self.parameters()).device
        self.to(device)
        vae.to(device)
        vae.eval()

        if freeze_vae:
            for p in vae.parameters():
                p.requires_grad = False

        optimiser = optimiser if optimiser is not None else optimizer
        if optimiser is None:
            optimiser = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        if schedule is None:
            schedule = self.cosine_schedule(num_steps=num_steps, device=device)
        else:
            schedule = _schedule_to_device(schedule, device)

        cfg = self.default_fit_config()
        cfg.update(fit_config)

        self._compile_state = {
            "vae": vae,
            "train_retrieval_bank": train_retrieval_bank,
            "val_retrieval_bank": val_retrieval_bank if val_retrieval_bank is not None else train_retrieval_bank,
            "schedule": schedule,
            "optimiser": optimiser,
            "device": device,
        }
        self._fit_config = cfg
        self._compiled = True
        return self

    def _require_compiled(self) -> Dict[str, Any]:
        if not self._compiled:
            raise RuntimeError("Call model.compile(...) before train_epoch(), eval_epoch(), or fit().")
        return self._compile_state

    @torch.no_grad()
    def build_self_condition(
        self,
        z_t: torch.Tensor,
        z_masked_input: torch.Tensor,
        m_latent_input: torch.Tensor,
        cond_tokens: torch.Tensor,
        t: torch.Tensor,
        schedule: SimpleNamespace,
        p_selfcond: float = 0.5,
    ) -> torch.Tensor:
        if torch.rand(1).item() < p_selfcond:
            zero_selfcond = torch.zeros_like(z_t)
            model_input_prev = torch.cat([z_t, z_masked_input, zero_selfcond, m_latent_input], dim=1)
            eps_hat_prev = self(model_input_prev, t, cond_tokens)
            return self.predict_x0(z_t, eps_hat_prev, t, schedule).detach()
        return torch.zeros_like(z_t)

    def diffusion_step(
        self,
        batch: Dict[str, Any],
        retrieval_bank: RetrievalBank,
        *,
        vae: nn.Module,
        schedule: SimpleNamespace,
        device: torch.device | str,
        noise_loss: str = "masked_mse",
        latent_loss: str = "l1",
        latent_loss_weight: float = 0.0,
        delta: float = 1.0,
        masked_weight: float = 3.0,
        use_min_snr: bool = True,
        min_snr_gamma: float = 5.0,
        use_self_condition: bool = True,
        p_selfcond: float = 0.5,
        cond_dropout_prob: float = 0.10,
        top_k: int = 5,
        max_keep: int = 3,
        diversity_threshold: float = 0.92,
        selective_temperature: float = 0.08,
        selective_sharpen_power: float = 2.0,
        selective_min_keep_weight: float = 0.05,
        retrieval_strength_cfg: Optional[Dict[str, Any]] = None,
        pad_multiple: int = 8,
        latent_context_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        device = torch.device(device)

        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)

        x, _ = padding(x, multiple=pad_multiple)
        y, _ = padding(y, multiple=pad_multiple)
        m, _ = padding(m, multiple=pad_multiple)

        with torch.no_grad():
            z_clean = encode2latentmean(vae, y)
            z_masked = encode2latentmean(vae, x)

        m_latent = F.interpolate(m, size=z_clean.shape[-2:], mode="nearest")
        z_context = z_masked * (1.0 - m_latent)

        z_masked_input = z_masked.clone()
        m_latent_input = m_latent.clone()

        if cond_dropout_prob > 0.0:
            keep = (
                (torch.rand(z_clean.shape[0], device=device) > cond_dropout_prob)
                .float()
                .view(-1, 1, 1, 1)
            )
            z_masked_input = z_masked_input * keep
            m_latent_input = m_latent_input * keep

        query_meta = retrieval_bank.batch_query_meta_from_batch(batch, batch_size=x.shape[0])

        with torch.no_grad():
            proposal = retrieval_bank.build_topk_candidates(
                z_context=z_context,
                m_latent=m_latent,
                device=device,
                top_k=top_k,
                max_keep=max_keep,
                diversity_threshold=diversity_threshold,
                query_meta=query_meta,
            )

        candidate_targets = proposal["candidate_targets"]
        candidate_scores = proposal["candidate_scores"]
        candidate_valid_mask = proposal["candidate_valid_mask"]

        if retrieval_strength_cfg is None:
            retrieval_strength = self.sample_retrieval_strength(batch_size=z_clean.shape[0], device=device)
        else:
            retrieval_strength = self.sample_retrieval_strength(
                batch_size=z_clean.shape[0],
                device=device,
                **retrieval_strength_cfg,
            )

        retrieval_cond = self.build_retrieval_condition(
            candidate_targets=candidate_targets,
            candidate_scores=candidate_scores,
            candidate_valid_mask=candidate_valid_mask,
            z_context=z_context,
            m_latent=m_latent,
            retrieval_strength=retrieval_strength,
            selective_temperature=selective_temperature,
            selective_sharpen_power=selective_sharpen_power,
            selective_min_keep_weight=selective_min_keep_weight,
        )
        cond_tokens = retrieval_cond["cond_tokens"]

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

        if use_self_condition:
            z0_selfcond = self.build_self_condition(
                z_t=z_t,
                z_masked_input=z_masked_input,
                m_latent_input=m_latent_input,
                cond_tokens=cond_tokens,
                t=t,
                schedule=schedule,
                p_selfcond=p_selfcond,
            )
        else:
            z0_selfcond = torch.zeros_like(z_clean)

        model_input = torch.cat([z_t, z_masked_input, z0_selfcond, m_latent_input], dim=1)
        eps_hat = self(model_input, t, cond_tokens)

        noise_loss_value = self.compute_noise_loss(
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

        z0_hat = self.predict_x0(z_t, eps_hat, t, schedule)

        if latent_loss_weight > 0.0:
            latent_loss_value, latent_gap_loss, latent_context_loss = self.compute_latent_loss(
                pred_latent=z0_hat,
                target_latent=z_clean,
                mask_latent=m_latent,
                latent_loss=latent_loss,
                context_weight=latent_context_weight,
            )
        else:
            latent_loss_value = torch.tensor(0.0, device=device)
            latent_gap_loss = torch.tensor(0.0, device=device)
            latent_context_loss = torch.tensor(0.0, device=device)

        total_loss = noise_loss_value + latent_loss_weight * latent_loss_value
        fusion_weights = retrieval_cond["fusion_weights"]
        fusion_entropy = -(fusion_weights * torch.log(fusion_weights + 1e-8)).sum(dim=1).mean()

        return {
            "loss": total_loss,
            "noise_loss": noise_loss_value.detach(),
            "latent_loss": latent_loss_value.detach(),
            "latent_gap_loss": latent_gap_loss.detach(),
            "latent_context_loss": latent_context_loss.detach(),
            "retrieval_strength_mean": retrieval_strength.mean().detach(),
            "fusion_weight_entropy": fusion_entropy.detach(),
        }

    @torch.no_grad()
    def infer_diffusion_latent(
        self,
        vae,
        schedule,
        x_masked,
        mask,
        device,
        retrieval_bank=None,
        num_steps=None,
        return_all_steps=False,
        use_self_conditioning=False,
        retrieval_strength=0.8,
        top_k=5,
        max_keep=3,
        diversity_threshold=0.92,
        selective_temperature=0.08,
        selective_sharpen_power=2.0,
        selective_min_keep_weight=0.05,
        query_meta=None,
        pad_multiple=8,
        noise_known_region=True,
    ):

        self.eval()
        vae.eval()

        device = torch.device(device)
        x_masked = x_masked.to(device)
        mask = mask.to(device)

        # move schedule to correct device
        schedule = SimpleNamespace(
            **{
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in vars(schedule).items()
            }
        )

        # use compiled retrieval bank if not manually passed
        if retrieval_bank is None:
            if hasattr(self, "_compiled") and self._compiled:
                retrieval_bank = self._compile_state.get("train_retrieval_bank", None)

        if retrieval_bank is None:
            raise ValueError(
                "retrieval_bank must be provided for RetrievalConditionedDiffusionUNet inference."
            )

        # encode masked spectrogram into latent space
        x_pad, pad_info = padding(x_masked, multiple=pad_multiple)
        m_pad, _ = padding(mask, multiple=pad_multiple)

        z_masked = encode2latentmean(vae, x_pad)

        m_latent = F.interpolate(
            m_pad,
            size=z_masked.shape[-2:],
            mode="nearest",
        )

        # mask convention:
        # m_latent == 1 in gap
        # m_latent == 0 in known context
        z_context = z_masked * (1.0 - m_latent)

        # build retrieval conditioning
        proposal = retrieval_bank.build_topk_candidates(
            z_context=z_context,
            m_latent=m_latent,
            device=device,
            top_k=top_k,
            max_keep=max_keep,
            diversity_threshold=diversity_threshold,
            query_meta=query_meta,
        )

        candidate_targets = proposal["candidate_targets"]
        candidate_scores = proposal["candidate_scores"]
        candidate_valid_mask = proposal["candidate_valid_mask"]

        if isinstance(retrieval_strength, (float, int)):
            retrieval_strength = torch.full(
                (z_masked.shape[0], 1, 1, 1),
                float(retrieval_strength),
                device=device,
            )
        else:
            retrieval_strength = retrieval_strength.to(device)

        retrieval_cond = self.build_retrieval_condition(
            candidate_targets=candidate_targets,
            candidate_scores=candidate_scores,
            candidate_valid_mask=candidate_valid_mask,
            z_context=z_context,
            m_latent=m_latent,
            retrieval_strength=retrieval_strength,
            selective_temperature=selective_temperature,
            selective_sharpen_power=selective_sharpen_power,
            selective_min_keep_weight=selective_min_keep_weight,
        )

        cond_tokens = retrieval_cond["cond_tokens"]

        # build timestep sequence
        total_steps = schedule.betas.shape[0]

        if num_steps is None:
            num_steps = total_steps

        if num_steps != total_steps:
            print(
                "Warning: using fewer inference steps than training steps. "
                "This is an approximate DDPM sampler. For safest results, use num_steps=1000."
            )

        timesteps = torch.linspace(
            total_steps - 1,
            0,
            steps=num_steps,
            device=device,
        ).long()

        # remove possible duplicate timesteps if num_steps is small
        timesteps = torch.unique_consecutive(timesteps)

        bsz = z_masked.shape[0]

        # initialise z_T
        unknown_noise = torch.randn_like(z_masked)
        known_noise = torch.randn_like(z_masked)

        t_start = torch.full(
            (bsz,),
            int(timesteps[0].item()),
            device=device,
            dtype=torch.long,
        )

        if noise_known_region:
            z_known_t = self.q_sample(
                z_0=z_masked,
                t=t_start,
                noise=known_noise,
                schedule=schedule,
            )
        else:
            z_known_t = z_masked

        # random gap, correctly noised known region
        z = m_latent * unknown_noise + (1.0 - m_latent) * z_known_t

        z0_selfcond = torch.zeros_like(z)
        all_steps = []

        # reverse diffusion loop
        for step_idx, t_scalar in enumerate(
            tqdm(timesteps, desc="Enhanced diffusion inference", leave=False)
        ):
            t_int = int(t_scalar.item())

            t = torch.full(
                (bsz,),
                t_int,
                device=device,
                dtype=torch.long,
            )

            model_input = torch.cat(
                [z, z_masked, z0_selfcond, m_latent],
                dim=1,
            )

            eps_hat = self(model_input, t, cond_tokens)

            z0_hat = self.predict_x0(
                z_t=z,
                eps_hat=eps_hat,
                t=t,
                schedule=schedule,
            )

            beta_t = extract(schedule.betas, t, z.shape)

            sqrt_one_minus_alpha_bar_t = extract(
                schedule.sqrt_one_minus_alpha_bars,
                t,
                z.shape,
            )

            sqrt_recip_alpha_t = extract(
                schedule.sqrt_recip_alphas,
                t,
                z.shape,
            )

            model_mean = sqrt_recip_alpha_t * (
                z - beta_t * eps_hat / (sqrt_one_minus_alpha_bar_t + 1e-8)
            )

            if t_int > 0:
                posterior_var_t = extract(
                    schedule.posterior_variance,
                    t,
                    z.shape,
                )

                z_prev = model_mean + torch.sqrt(
                    posterior_var_t.clamp(min=1e-20)
                ) * torch.randn_like(z)
            else:
                z_prev = model_mean

            # preserve known region at the next timestep, not current timestep
            if step_idx < len(timesteps) - 1:
                next_t_int = int(timesteps[step_idx + 1].item())
            else:
                next_t_int = 0

            if next_t_int > 0 and noise_known_region:
                next_t = torch.full(
                    (bsz,),
                    next_t_int,
                    device=device,
                    dtype=torch.long,
                )

                z_known_next = self.q_sample(
                    z_0=z_masked,
                    t=next_t,
                    noise=known_noise,
                    schedule=schedule,
                )
            else:
                z_known_next = z_masked

            z = m_latent * z_prev + (1.0 - m_latent) * z_known_next

            if use_self_conditioning:
                z0_selfcond = z0_hat.detach()
            else:
                z0_selfcond = torch.zeros_like(z)

            if return_all_steps:
                all_steps.append(z.detach().cpu())

        # 6. decode final latent
        x_hat_pad = decode_from_latent(vae, z)
        x_hat = unpadding(x_hat_pad, pad_info)

        return {
            "x_hat": x_hat,
            "z_hat": z,
            "m_latent": m_latent,
            "cond_tokens": cond_tokens,
            "retrieval_cond": retrieval_cond,
            "candidate_indices": proposal.get("candidate_indices", None),
            "raw_top_scores": proposal.get("raw_top_scores", None),
            "raw_top_idx": proposal.get("raw_top_idx", None),
            "all_steps": all_steps if return_all_steps else None,
        }

    def train_epoch(
        self,
        dataloader: Iterable[Dict[str, Any]],
        retrieval_bank: Optional[RetrievalBank] = None,
        **overrides: Any,
    ) -> Dict[str, float]:
        state = self._require_compiled()
        vae = state["vae"]
        schedule = state["schedule"]
        optimiser = state["optimiser"]
        device = state["device"]
        retrieval_bank = retrieval_bank or state["train_retrieval_bank"]

        cfg = dict(self._fit_config)
        cfg.update(overrides)

        self.train()
        vae.eval()

        running = {
            "loss": 0.0,
            "noise_loss": 0.0,
            "latent_loss": 0.0,
            "retrieval_strength_mean": 0.0,
            "fusion_weight_entropy": 0.0,
        }
        n_batches = 0

        use_amp = bool(cfg.pop("use_amp")) and _device_type(device) == "cuda"
        grad_clip = cfg.pop("grad_clip")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for batch in tqdm(dataloader, desc="train diffusion", leave=False):
            optimiser.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=_device_type(device), enabled=use_amp):
                out = self.diffusion_step(
                    batch=batch,
                    retrieval_bank=retrieval_bank,
                    vae=vae,
                    schedule=schedule,
                    device=device,
                    **cfg,
                )

            scaler.scale(out["loss"]).backward()

            if grad_clip is not None:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip)

            scaler.step(optimiser)
            scaler.update()

            for k in running:
                running[k] += float(out[k].item())
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    @torch.no_grad()
    def eval_epoch(
        self,
        dataloader: Iterable[Dict[str, Any]],
        retrieval_bank: Optional[RetrievalBank] = None,
        **overrides: Any,
    ) -> Dict[str, float]:
        state = self._require_compiled()
        vae = state["vae"]
        schedule = state["schedule"]
        device = state["device"]
        retrieval_bank = retrieval_bank or state["val_retrieval_bank"]

        cfg = dict(self._fit_config)
        cfg.update(
            {
                "use_self_condition": False,
                "p_selfcond": 0.0,
                "cond_dropout_prob": 0.0,
            }
        )
        cfg.update(overrides)

        self.eval()
        vae.eval()

        running = {
            "loss": 0.0,
            "noise_loss": 0.0,
            "latent_loss": 0.0,
            "retrieval_strength_mean": 0.0,
            "fusion_weight_entropy": 0.0,
        }
        n_batches = 0

        use_amp = bool(cfg.pop("use_amp")) and _device_type(device) == "cuda"
        cfg.pop("grad_clip", None)

        for batch in tqdm(dataloader, desc="eval diffusion", leave=False):
            with torch.amp.autocast(device_type=_device_type(device), enabled=use_amp):
                out = self.diffusion_step(
                    batch=batch,
                    retrieval_bank=retrieval_bank,
                    vae=vae,
                    schedule=schedule,
                    device=device,
                    **cfg,
                )

            for k in running:
                running[k] += float(out[k].item())
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    def fit(
        self,
        train_loader: Iterable[Dict[str, Any]],
        val_loader: Iterable[Dict[str, Any]],
        *,
        n_epochs: int,
        checkpoint_dir: str | Path,
        history_path: str | Path,
        monitor: str = "val_noise_loss",
        mode: str = "min",
        patience: int = 10,
        min_delta: float = 1e-4,
        save_best_after_epoch: int = 1,
        use_scheduler: bool = True,
        scheduler_type: str = "plateau",
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 3,
        scheduler_min_lr: float = 1e-7,
        resume_checkpoint_path: Optional[str | Path] = None,
        load_history: bool = False,
        **fit_overrides: Any,
    ) -> Dict[str, Any]:
        """
        full training loop using objects attached in compile().
        """
        state = self._require_compiled()
        optimiser = state["optimiser"]
        device = state["device"]

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_cls = _require_external("ModelCheckpoint")
        manager = checkpoint_cls(
            checkpoint_dir=checkpoint_dir,
            monitor=monitor,
            mode=mode,
            patience=patience,
            min_delta=min_delta,
            save_best_after_epoch=save_best_after_epoch,
            verbose=True,
        )

        history_keys = [
            "epoch",
            "lr",
            "train_loss",
            "train_noise_loss",
            "train_latent_loss",
            "train_retrieval_strength_mean",
            "train_fusion_weight_entropy",
            "val_loss",
            "val_noise_loss",
            "val_latent_loss",
            "val_retrieval_strength_mean",
            "val_fusion_weight_entropy",
        ]
        history: Dict[str, List[Any]] = {k: [] for k in history_keys}
        start_epoch = 1

        if resume_checkpoint_path is not None:
            resume_checkpoint_path = Path(resume_checkpoint_path)
            if resume_checkpoint_path.exists():
                ckpt = torch.load(resume_checkpoint_path, map_location=device)
                print("checkpoint keys:", ckpt.keys())
                print("saved epoch:", ckpt.get("epoch", None))
                print("best score:", ckpt.get("best_score", None))
                print("best epoch:", ckpt.get("best_epoch", None))
                if "metrics" in ckpt:
                    print("checkpoint val_noise_loss:", ckpt["metrics"].get("val_noise_loss", None))
                    print("checkpoint val_loss:", ckpt["metrics"].get("val_loss", None))

                self.load_state_dict(ckpt["model_state_dict"])

                if "optimiser_state_dict" in ckpt:
                    optimiser.load_state_dict(ckpt["optimiser_state_dict"])

                start_epoch = int(ckpt.get("epoch", 0)) + 1
                manager.best_score = ckpt.get("best_score", manager.best_score)
                manager.best_epoch = ckpt.get("best_epoch", manager.best_epoch)
            else:
                print(f"resume checkpoint not found: {resume_checkpoint_path}")

        if load_history and history_path.exists():
            old_history_df = pd.read_csv(history_path)
            missing_cols = [k for k in history_keys if k not in old_history_df.columns]
            if len(missing_cols) == 0:
                history = {col: old_history_df[col].tolist() for col in history_keys}
            else:
                print(f"history exists but missing columns {missing_cols}; starting a fresh history dict")

        scheduler = None
        if use_scheduler:
            if scheduler_type == "plateau":
                scheduler = ReduceLROnPlateau(
                    optimiser,
                    mode=mode,
                    factor=scheduler_factor,
                    patience=scheduler_patience,
                    min_lr=scheduler_min_lr,
                )
            elif scheduler_type == "cosine":
                scheduler = CosineAnnealingLR(
                    optimiser,
                    T_max=n_epochs,
                    eta_min=scheduler_min_lr,
                )
            else:
                raise ValueError(f"unsupported scheduler_type: {scheduler_type}")

        cfg = dict(self._fit_config)
        cfg.update(fit_overrides)

        for epoch in tqdm(range(start_epoch, n_epochs + 1), desc="training diffusion"):
            train_metrics = self.train_epoch(
                dataloader=train_loader,
                retrieval_bank=state["train_retrieval_bank"],
                **cfg,
            )
            val_metrics = self.eval_epoch(
                dataloader=val_loader,
                retrieval_bank=state["val_retrieval_bank"],
                **cfg,
            )

            if scheduler is not None:
                if scheduler_type == "plateau":
                    if monitor == "val_loss":
                        scheduler.step(val_metrics["loss"])
                    elif monitor == "val_noise_loss":
                        scheduler.step(val_metrics["noise_loss"])
                    else:
                        scheduler.step(val_metrics["loss"])
                else:
                    scheduler.step()

            current_lr = get_lr(optimiser)
            epoch_record = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_noise_loss": train_metrics["noise_loss"],
                "train_latent_loss": train_metrics["latent_loss"],
                "train_retrieval_strength_mean": train_metrics["retrieval_strength_mean"],
                "train_fusion_weight_entropy": train_metrics["fusion_weight_entropy"],
                "val_loss": val_metrics["loss"],
                "val_noise_loss": val_metrics["noise_loss"],
                "val_latent_loss": val_metrics["latent_loss"],
                "val_retrieval_strength_mean": val_metrics["retrieval_strength_mean"],
                "val_fusion_weight_entropy": val_metrics["fusion_weight_entropy"],
            }

            for key in history_keys:
                history[key].append(epoch_record[key])

            pd.DataFrame(history).to_csv(history_path, index=False)

            print(f"epoch {epoch:02d}")
            print(f"learning rate:               {current_lr:.8e}")
            print(f"train loss:                  {train_metrics['loss']:.6f}")
            print(f"val loss:                    {val_metrics['loss']:.6f}")
            print(f"train noise loss:            {train_metrics['noise_loss']:.6f}")
            print(f"val noise loss:              {val_metrics['noise_loss']:.6f}")
            print(f"train latent loss:           {train_metrics['latent_loss']:.6f}")
            print(f"val latent loss:             {val_metrics['latent_loss']:.6f}")
            print(f"train retrieval strength:    {train_metrics['retrieval_strength_mean']:.4f}")
            print(f"val retrieval strength:      {val_metrics['retrieval_strength_mean']:.4f}")
            print(f"train fusion entropy:        {train_metrics['fusion_weight_entropy']:.4f}")
            print(f"val fusion entropy:          {val_metrics['fusion_weight_entropy']:.4f}")
            print("-" * 60)

            manager.step(
                epoch=epoch,
                metrics=epoch_record,
                model=self,
                optimiser=optimiser,
            )

            if manager.should_stop:
                print(f"early stopping triggered at epoch {epoch}")
                break

        return {
            "history": history,
            "best_score": manager.best_score,
            "best_epoch": manager.best_epoch,
            "checkpoint_dir": checkpoint_dir,
        }
