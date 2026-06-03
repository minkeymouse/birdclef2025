"""paper_update — patch paper Q-test section with actual LB results.

Replaces the `[Numerical results pending; ...]` placeholder in
`paper/exp_current.tex` with a numerical table once lb_registry has scores.
Idempotent: skips if placeholder is already gone.
"""
from __future__ import annotations
from pathlib import Path

from ._common import ANCHOR_LB, ROOT, get_lb_score

PAPER_TEX = ROOT / "paper/exp_current.tex"
PLACEHOLDER = "[Numerical results pending; will be inserted when LB scores return.]"


def fmt_delta(lb: float | None) -> str:
    if lb is None:
        return "pending"
    return f"{lb:.4f} ($\\Delta${lb - ANCHOR_LB:+.4f})"


def main():
    text = PAPER_TEX.read_text()
    if PLACEHOLDER not in text:
        print("Already updated or placeholder not found.", flush=True)
        return

    lb_v26 = get_lb_score("v26")
    lb_v27 = get_lb_score("v27")
    lb_v28 = get_lb_score("v28")
    lb_v30 = get_lb_score("v30")

    if all(x is None for x in [lb_v26, lb_v27, lb_v28, lb_v30]):
        print("No LB results yet. Skipping paper update.", flush=True)
        return

    table = (
        "Numerical results:\n\n"
        "\\begin{tabular}{lcccc}\n"
        "Test & v26 (Q1) & v27 (Q2) & v28 (Q3) & v30 (exp175) \\\\\n"
        f"LB ($\\Delta$ vs {ANCHOR_LB:.3f}) & {fmt_delta(lb_v26)} & {fmt_delta(lb_v27)} & "
        f"{fmt_delta(lb_v28)} & {fmt_delta(lb_v30)} \\\\\n"
        "\\end{tabular}\n"
    )
    PAPER_TEX.write_text(text.replace(PLACEHOLDER, table))
    print(
        f"Paper updated. v26={lb_v26}, v27={lb_v27}, v28={lb_v28}, v29={lb_v30}",
        flush=True,
    )


if __name__ == "__main__":
    main()
