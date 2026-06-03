from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sidecar_src.utils.audio import load_audio_segment
from sidecar_src.utils.features import make_feature


class SoundscapeWindowDataset(Dataset):
    def __init__(
        self,
        meta: pd.DataFrame,
        targets: np.ndarray | None,
        cfg: dict,
        target_masks: np.ndarray | None = None,
    ):
        self.meta = meta.reset_index(drop=True).copy()
        self.targets = targets
        self.target_masks = target_masks
        self.cfg = cfg
        self.sr = int(cfg["audio"]["sample_rate"])
        if targets is not None:
            assert len(self.meta) == targets.shape[0]
        if target_masks is not None:
            assert targets is not None
            assert target_masks.shape == targets.shape

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.meta.iloc[idx]
        wav = load_audio_segment(
            Path(row["path"]),
            float(row["context_start_sec"]),
            float(row["context_end_sec"]),
            self.sr,
        )
        image = make_feature(wav, self.cfg)
        item: dict[str, torch.Tensor | str] = {
            "image": torch.from_numpy(image),
            "row_id": str(row["row_id"]),
        }
        if self.targets is not None:
            target = self.targets[idx].astype(np.float32, copy=False)
            item["target"] = torch.from_numpy(target)
            if self.target_masks is None:
                item["target_mask"] = torch.ones_like(item["target"])
            else:
                mask = self.target_masks[idx].astype(np.float32, copy=False)
                item["target_mask"] = torch.from_numpy(mask)
        return item
