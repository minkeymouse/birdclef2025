"""SED model — single config-driven DistilledSED.

Replaces 10 duplicate copies across exp169...exp175. Init behavior is selected
by SedConfig fields, so the same class can express:
  - Tucker canonical (drop_path_rate only, xavier init)
  - exp169 legacy (extra drop_rate, default init)
  - any future variant
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import timm

from .config import SedConfig


class MelExtractor(nn.Module):
    """Mel + AmplitudeToDB + per-sample z-score (Tucker spec)."""

    def __init__(self, cfg: SedConfig):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.SR, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
            n_mels=cfg.N_MELS, f_min=cfg.FMIN, f_max=cfg.FMAX,
            power=2.0, center=True,
        )
        self.adb = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=cfg.AMPLITUDE_TO_DB_TOP_DB
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        m = self.adb(self.mel(x))
        mean = m.mean(dim=(1, 2), keepdim=True)
        std = m.std(dim=(1, 2), keepdim=True).clamp(min=1e-3)
        m = (m - mean) / std
        return m.unsqueeze(1)  # (B, 1, n_mels, T)


class GeMFreqPool(nn.Module):
    """Generalized-mean pooling over frequency dim. Learnable p starts at p_init."""

    def __init__(self, p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.full((1,), float(p_init)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = x.mean(dim=2)  # mean over frequency
        return x.pow(1.0 / self.p)


class SpecAug(nn.Module):
    """SpecAugment with config-driven mask counts (Tucker: 1 freq + 2 time)."""

    def __init__(self, cfg: SedConfig):
        super().__init__()
        self.fm = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=cfg.SPEC_FREQ_MASK_PARAM
        )
        self.tm = torchaudio.transforms.TimeMasking(
            time_mask_param=cfg.SPEC_TIME_MASK_PARAM
        )
        self.nf = cfg.SPEC_NUM_FREQ_MASKS
        self.nt = cfg.SPEC_NUM_TIME_MASKS

    def forward(self, x):
        for _ in range(self.nf):
            x = self.fm(x)
        for _ in range(self.nt):
            x = self.tm(x)
        return x


class DistilledSED(nn.Module):
    """Single SED model used by all experiments. Config drives init choices.

    Architecture (Tucker spec):
      mel → backbone → [GeMFreq pool → bottleneck → att-pooled clip logits]
                     → [GAP+Linear → Perch distill embedding]
    Stop-gradient between backbone and SED head: backbone trains via distill only.
    """

    def __init__(self, cfg: SedConfig, n_cls: int = 234):
        super().__init__()
        self.cfg = cfg
        self.mel = MelExtractor(cfg)
        self.spec_aug = SpecAug(cfg)

        # Build backbone kwargs: only pass drop_rate if explicitly set.
        bb_kwargs = dict(
            in_chans=1,
            num_classes=0,
            global_pool="",
            drop_path_rate=cfg.BACKBONE_DROP_PATH_RATE,
        )
        if cfg.BACKBONE_DROP_RATE is not None:
            bb_kwargs["drop_rate"] = cfg.BACKBONE_DROP_RATE
        self.backbone = timm.create_model(
            cfg.BACKBONE_NAME, pretrained=cfg.BACKBONE_PRETRAINED, **bb_kwargs
        )
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 1, cfg.N_MELS, 100))
            C = feat.shape[1]
        self.feat_dim = C

        self.gem_freq = GeMFreqPool(p_init=cfg.GEM_FREQ_P_INIT)
        self.bottleneck = nn.Sequential(
            nn.Dropout(cfg.BOTTLENECK_DROPOUT_1),
            nn.Linear(C, cfg.HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.BOTTLENECK_DROPOUT_2),
        )
        self.att = nn.Conv1d(cfg.HIDDEN_DIM, n_cls, kernel_size=1, bias=cfg.ATT_CLA_BIAS)
        self.cla = nn.Conv1d(cfg.HIDDEN_DIM, n_cls, kernel_size=1, bias=cfg.ATT_CLA_BIAS)
        if cfg.ATT_CLA_INIT == "xavier_uniform":
            nn.init.xavier_uniform_(self.att.weight)
            nn.init.xavier_uniform_(self.cla.weight)
            if cfg.ATT_CLA_BIAS:
                self.att.bias.data.fill_(0.0)
                self.cla.bias.data.fill_(0.0)
        # else: rely on PyTorch default (Kaiming-like for Conv1d)

        if cfg.USE_PERCH_DISTILL:
            self.distill_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(C, cfg.PERCH_EMBED_DIM),
            )
        else:
            self.distill_head = None

    def forward(self, x: torch.Tensor, training_aug: bool = False):
        # Mel + SpecAug must run in fp32 (AmplitudeToDB underflows in fp16)
        with torch.amp.autocast("cuda", enabled=False):
            m = self.mel(x.float())
            if training_aug and self.training:
                m = self.spec_aug(m)
        h = self.backbone(m)

        distill_emb = self.distill_head(h) if self.distill_head is not None else None

        # Stop-gradient: SED head doesn't propagate to backbone
        h_cls = h.detach() if self.cfg.STOP_GRAD_TO_BACKBONE else h
        h_cls = self.gem_freq(h_cls)            # (B, C, T)
        h_cls = h_cls.transpose(1, 2)           # (B, T, C)
        h_cls = self.bottleneck(h_cls)          # (B, T, hidden)
        h_cls = h_cls.transpose(1, 2)           # (B, hidden, T)

        att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise_logits = self.cla(h_cls)
        clip_logits = (att * framewise_logits).sum(dim=2)
        return clip_logits, framewise_logits, distill_emb


def softauc_loss(logits: torch.Tensor, y: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """Babych 2025 1st place SoftAUCLoss.

    Pairwise log-loss over (positive, negative) pairs within a batch.
    For each class c, compute:
        L_c = mean_{i: y_i=1, j: y_j=0} log(1 + exp((s_j - s_i) / T))
    Equivalent to direct AUC optimization (matches macro-AUC metric exactly).

    Implementation note: torch.cdist of scores would explode memory; use
    broadcast over batch, mask by y/(1-y) outer product, mean over valid pairs.
    """
    s = logits / max(T, 1e-6)  # (B, C)
    y_pos = y                  # (B, C) 1 where positive
    y_neg = 1.0 - y            # (B, C) 1 where negative
    # diff[i, j, c] = s[i, c] - s[j, c]; we want positives (y_i=1) > negatives (y_j=0)
    s_diff = s.unsqueeze(1) - s.unsqueeze(0)  # (B, B, C)
    pair_mask = y_pos.unsqueeze(1) * y_neg.unsqueeze(0)  # (B, B, C); 1 where (i is pos, j is neg)
    pair_loss = F.softplus(-s_diff)  # log(1 + exp(-(s_pos - s_neg)))
    # mean over valid pairs (per class), then mean over classes
    per_class_loss = (pair_loss * pair_mask).sum(dim=(0, 1)) / pair_mask.sum(dim=(0, 1)).clamp(min=1.0)
    return per_class_loss.mean()


def hybrid_loss(
    clip_logits: torch.Tensor,
    framewise_logits: torch.Tensor,
    distill_emb: torch.Tensor | None,
    perch_emb: torch.Tensor | None,
    y: torch.Tensor,
    cfg: SedConfig,
    is_pseudo: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
):
    """Tucker hybrid loss with optional pseudo-mask.

    cls_loss = w_clip * BCE(clip_logits, y) + w_frame * BCE(frame_max, y)
    distill_loss = MSE(distill_emb, perch_emb), masked by is_pseudo (=0 for pseudo)
    total = cls_loss + α * distill_loss

    If cfg.LOSS_FAMILY == 'softauc', replaces both BCE terms with SoftAUC.
    """
    use_softauc = getattr(cfg, "LOSS_FAMILY", "bce") == "softauc"
    if use_softauc:
        T = getattr(cfg, "SOFTAUC_TEMPERATURE", 1.0)
        bce_clip = softauc_loss(clip_logits, y, T=T)
        frame_max = framewise_logits.max(dim=2).values
        bce_frame = softauc_loss(frame_max, y, T=T)
    else:
        if pos_weight is not None:
            bce_clip = F.binary_cross_entropy_with_logits(clip_logits, y, pos_weight=pos_weight)
            frame_max = framewise_logits.max(dim=2).values
            bce_frame = F.binary_cross_entropy_with_logits(frame_max, y, pos_weight=pos_weight)
        else:
            bce_clip = F.binary_cross_entropy_with_logits(clip_logits, y)
            frame_max = framewise_logits.max(dim=2).values
            bce_frame = F.binary_cross_entropy_with_logits(frame_max, y)
    cls_loss = cfg.BCE_CLIP_WEIGHT * bce_clip + cfg.BCE_FRAME_MAX_WEIGHT * bce_frame

    if distill_emb is not None and perch_emb is not None:
        if is_pseudo is not None and is_pseudo.any():
            mse_per = F.mse_loss(distill_emb, perch_emb, reduction="none").mean(dim=1)
            valid_mask = (~is_pseudo).float()
            denom = valid_mask.sum().clamp(min=1.0)
            distill_loss = (mse_per * valid_mask).sum() / denom
        else:
            distill_loss = F.mse_loss(distill_emb, perch_emb)
    else:
        distill_loss = torch.tensor(0.0, device=clip_logits.device)

    total = cls_loss + cfg.ALPHA_DISTILL * distill_loss
    return total, bce_clip.item(), bce_frame.item(), float(distill_loss.item())
