"""memory_update — write Q-test outcomes to user memory dir.

Generates `memory/project_q_test_results.md` summarizing Q1/Q2/Q3 outcomes +
exp175 outcome (when present), with pseudo strategy decisions.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from ._common import ANCHOR_LB, classify_lb_delta, get_lb_score

MEMORY_DIR = Path.home() / ".claude/projects/-data-birdclef2026/memory"


@dataclass(frozen=True)
class QResult:
    label: str
    subset: str
    n_classes: int


Q_RESULTS = (
    QResult("Q1 (v26)", "Insecta", 28),
    QResult("Q2 (v27)", "Amphibia", 35),
    QResult("Q3 (v28)", "rare n_train_audio<5", 42),
)


def category_recommendation(subset: str, category: str, delta: float) -> str:
    if category.startswith("regression"):
        if "strong" in category:
            return (
                f"- **{subset}**: pseudo HIGH PRIORITY. Strong contribution to LB. "
                f"Generate aggressive pseudo for these classes (high threshold, oversample)."
            )
        return f"- **{subset}**: pseudo modest priority. Conservative dose."
    if category == "neutral":
        return f"- **{subset}**: pseudo effect uncertain. Skip or low-dose test."
    if category == "improvement_mild":
        return f"- **{subset}**: EXCLUDE from pseudo (mildly hurting LB)."
    if category == "improvement_strong":
        return (
            f"- **{subset}**: SUPPRESS in production (loss weight ↓ or zero-out). "
            f"Strong signal of class-level val-LB anti-correlation."
        )
    return f"- **{subset}**: unclassified delta {delta:+.4f}"


def main():
    lb_q1 = get_lb_score("v26")
    lb_q2 = get_lb_score("v27")
    lb_q3 = get_lb_score("v28")
    lb_exp175 = get_lb_score("v30")

    if all(x is None for x in [lb_q1, lb_q2, lb_q3]):
        print("No Q-test results yet.", flush=True)
        return

    out_path = MEMORY_DIR / "project_q_test_results.md"

    lines = [
        "---",
        "name: Q-test results (2026-05-08)",
        "description: Q1/Q2/Q3 Insecta/Amphibia/rare-class subset diagnostic LB results + pseudo strategy decisions",
        "type: project",
        "---",
        "",
        "# Q-test results (2026-05-08)",
        "",
        f"Anchor (v5): LB {ANCHOR_LB:.4f}. Stochasticity ±0.002.",
        "",
        "| Q | Subset | Cls dropped | LB | Δ | Classification |",
        "|---|---|---|---|---|---|",
    ]

    rows: list[tuple[QResult, float | None]] = [
        (Q_RESULTS[0], lb_q1),
        (Q_RESULTS[1], lb_q2),
        (Q_RESULTS[2], lb_q3),
    ]
    for q, lb in rows:
        if lb is None:
            lines.append(f"| {q.label} | {q.subset} | {q.n_classes} | pending | — | — |")
        else:
            delta = lb - ANCHOR_LB
            cat = classify_lb_delta(delta)
            lines.append(f"| {q.label} | {q.subset} | {q.n_classes} | {lb:.4f} | {delta:+.4f} | {cat} |")

    lines.extend(["", "## Pseudo strategy decisions", ""])
    for q, lb in rows:
        if lb is None:
            continue
        delta = lb - ANCHOR_LB
        cat = classify_lb_delta(delta)
        lines.append(category_recommendation(q.subset, cat, delta))

    if lb_exp175 is not None:
        delta = lb_exp175 - ANCHOR_LB
        lines.extend([
            "",
            f"## exp175 (v29) result: {lb_exp175:.4f} (Δ{delta:+.4f})",
        ])
        if delta >= 0.002:
            lines.append("→ Silent drift fix mattered. Multi-seed ensemble next priority.")
        elif delta <= -0.002:
            lines.append("→ Silent drift fix didn't help; possibly hurt. Investigate ckpt diff vs exp169.")
        else:
            lines.append("→ Within noise. Drift wasn't the bottleneck. Pivot to per-fold SS (exp176).")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}", flush=True)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
