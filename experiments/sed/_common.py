"""sed/_common — shared constants, paths, Kaggle helpers.

Single source of truth for:
- Repository root and notebook paths
- Kaggle slug and dataset references
- Anchor LB baseline (production = eos8-verbatim 0.950)
- Kaggle CLI helpers (try_submit, kernel_status, get_pushed_version)
- Eval helpers (macro_auc)

NOTE: this module + the lb_poller/lb_processor/deploy* orchestration target the LEGACY mattia-fork lineage
(superseded by the eos8 pipeline). Kept for reference; the current deploy path is the top-level
experiments/exp189_patch_notebook.py against eos8-phase4.
"""
from __future__ import annotations
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable

# ── Project paths ─────────────────────────────────────────────────────
ROOT = Path("/data/birdclef2026")
DATA_DIR = ROOT / "data/birdclef-2026"
NB_DIR = ROOT / "notebooks/birdclef-2026-mattia-fork"
NB_PATH = NB_DIR / "notebook.ipynb"
META_PATH = NB_DIR / "kernel-metadata.json"
LOG_DIR = ROOT / "experiments/_scratch_logs"
REGISTRY_YAML = ROOT / "experiments/lb_registry.yaml"

# ── Kaggle ────────────────────────────────────────────────────────────
KERNEL_SLUG = "ultimatumgame/birdclef-2026-mattia-fork"
COMP_SLUG = "birdclef-2026"
KAGGLE_TOKEN_ENV = "KAGGLE_API_TOKEN"

PERCH_META_DATASET = "jaejohn/perch-meta"
PERCH_ONNX_DATASET = "rishikeshjani/perch-onnx-for-birdclef-2026"

# Standard fixed dataset_sources for our kernel
STANDARD_DATASETS = [PERCH_META_DATASET, PERCH_ONNX_DATASET]

EXP169_SED_DATASET = "ultimatumgame/exp169-distilled-sed"
EXP175_SED_DATASET = "ultimatumgame/bc2026-exp175-sed"

# ── Anchor ────────────────────────────────────────────────────────────
# Production never-fall-below anchor = eos8-verbatim LB 0.950 (used by lb_processor for delta classification).
# (Legacy own-weights SED anchor was c4df217 = LB 0.938, pre-eos8 — kept for reference only.)
ANCHOR_COMMIT = "c4df217"
ANCHOR_LB = 0.950
LB_NOISE_BAND = 0.002

# ── Kaggle CLI helpers ────────────────────────────────────────────────


def kaggle_env() -> dict:
    """Return os.environ — caller must set KAGGLE_API_TOKEN beforehand."""
    return os.environ.copy()


def try_submit(version: int, msg: str, *, slug: str = KERNEL_SLUG,
               competition: str = COMP_SLUG, output_file: str = "submission.csv",
               timeout_s: int = 120) -> tuple[bool, str]:
    """Attempt to submit a Kaggle kernel version.

    Returns (success, output_tail). Empirically:
      - 400 Bad Request without -f → CLI bug (must specify output file)
      - 400 Bad Request with -f → version still re-running on Kaggle (rarer)
      - 200 → submitted successfully (output is empty on success in newer CLI)
      - 401 → auth failed (refresh token)

    NOTE: Kaggle CLI 2.0.1 silently ignores submit when -f is missing on code
    competitions; we always pass -f submission.csv. Verify success by querying
    `kaggle competitions submissions` afterwards.
    """
    proc = subprocess.run(
        ["uv", "run", "kaggle", "competitions", "submit",
         "-c", competition, "-f", output_file,
         "-k", slug, "-v", str(version),
         "-m", msg],
        env=kaggle_env(), capture_output=True, text=True, timeout=timeout_s,
    )
    out = proc.stdout + proc.stderr
    # Newer CLI returns empty stdout on success
    if proc.returncode == 0 and "400" not in out and "401" not in out:
        return True, out[-300:] or f"submitted v{version}"
    if "400" in out or "Bad Request" in out:
        return False, "still running (400)"
    if "Successfully submitted" in out or "submitted" in out.lower():
        return True, out[-300:]
    if "401" in out or "Unauthorized" in out:
        return False, "auth failed (401)"
    return False, out[-300:]


def kernel_status(slug: str = KERNEL_SLUG, timeout_s: int = 30) -> str:
    """Returns latest version status: 'running', 'complete', 'error', 'unknown'."""
    proc = subprocess.run(
        ["uv", "run", "kaggle", "kernels", "status", slug],
        env=kaggle_env(), capture_output=True, text=True, timeout=timeout_s,
    )
    out = (proc.stdout + proc.stderr).lower()
    if "running" in out:
        return "running"
    if "complete" in out and "incomplete" not in out:
        return "complete"
    if "error" in out:
        return "error"
    return "unknown"


def push_kernel(*, cwd: Path = NB_DIR) -> int | None:
    """Push current notebook. Returns Kaggle version number from response, or None."""
    proc = subprocess.run(
        ["uv", "run", "kaggle", "kernels", "push"],
        cwd=cwd, env=kaggle_env(), capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    print(out[-400:], flush=True)
    m = re.search(r"[Kk]ernel version (\d+) successfully pushed", out)
    if m:
        v = int(m.group(1))
        # Persist for downstream consumers
        (cwd / "_last_pushed_version.txt").write_text(str(v))
        return v
    return None


def get_last_pushed_version(*, cwd: Path = NB_DIR) -> int | None:
    """Read version saved by push_kernel()."""
    p = cwd / "_last_pushed_version.txt"
    if p.exists():
        try:
            return int(p.read_text().strip())
        except ValueError:
            return None
    return None


def reset_notebook_to_anchor() -> None:
    """git checkout the v5 anchor commit's notebook.ipynb."""
    proc = subprocess.run(
        ["git", "checkout", ANCHOR_COMMIT, "--",
         "notebooks/birdclef-2026-mattia-fork/notebook.ipynb"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git checkout failed: {proc.stderr}")


# ── Eval helpers ──────────────────────────────────────────────────────


def macro_auc(Y_true, Y_score) -> float:
    """Macro-averaged ROC AUC over classes with at least 1 positive AND 1 negative.

    Lazy-imports sklearn to keep this module light.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in range(Y_true.shape[1]):
        n_pos = Y_true[:, c].sum()
        if 0 < n_pos < len(Y_true):
            try:
                aucs.append(roc_auc_score(Y_true[:, c], Y_score[:, c]))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0


def per_class_auc(Y_true, Y_score):
    """Per-class ROC AUC, NaN where undefined."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    aucs = np.full(Y_true.shape[1], np.nan)
    for c in range(Y_true.shape[1]):
        n_pos = Y_true[:, c].sum()
        if 0 < n_pos < len(Y_true):
            try:
                aucs[c] = roc_auc_score(Y_true[:, c], Y_score[:, c])
            except Exception:
                pass
    return aucs


# ── lb_registry helpers ───────────────────────────────────────────────


def get_lb_score(version_id: str) -> float | None:
    """Read LB score for `vXX` entry in lb_registry yaml.

    Splits on entry boundaries to extract this version's full block, then
    scans for `    lb: <float>`. Skips `lb: pending`.
    """
    text = REGISTRY_YAML.read_text()
    # Split on entry boundaries (preserve via lookahead)
    entries = re.split(r"(?=^  - id: v\d+)", text, flags=re.MULTILINE)
    target = None
    for ent in entries:
        if re.match(rf"  - id: {version_id}\s*\n", ent):
            target = ent
            break
    if target is None:
        return None
    score_m = re.search(r"^    lb: ([\d.]+)\s*$", target, re.MULTILINE)
    if not score_m:
        return None
    try:
        return float(score_m.group(1))
    except ValueError:
        return None


def classify_lb_delta(delta: float) -> str:
    """Categorize LB delta vs anchor (NOISE_BAND=±0.002)."""
    if delta < -0.005:
        return "regression_strong"
    if delta < -LB_NOISE_BAND:
        return "regression_mild"
    if delta > 0.005:
        return "improvement_strong"
    if delta > LB_NOISE_BAND:
        return "improvement_mild"
    return "neutral"
