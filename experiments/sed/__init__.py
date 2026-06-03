"""experiments.sed — canonical SED training + LB orchestration package.

Refactored 2026-05-08: consolidated ~5,700 LOC of fork-based scripts into
~3,200 LOC with shared `_common.py` for paths, Kaggle helpers, and eval
metrics.

Public re-exports for convenience (most-common):
"""
from ._common import (
    ANCHOR_COMMIT,
    ANCHOR_LB,
    KERNEL_SLUG,
    NB_PATH,
    ROOT,
    classify_lb_delta,
    get_lb_score,
    macro_auc,
    per_class_auc,
    reset_notebook_to_anchor,
    try_submit,
)

__all__ = [
    "ANCHOR_COMMIT",
    "ANCHOR_LB",
    "KERNEL_SLUG",
    "NB_PATH",
    "ROOT",
    "classify_lb_delta",
    "get_lb_score",
    "macro_auc",
    "per_class_auc",
    "reset_notebook_to_anchor",
    "try_submit",
]
