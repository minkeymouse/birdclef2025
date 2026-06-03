"""lb_processor — auto-fill lb_registry pending entries when scores arrive.

Maps submission descriptions in `submissions_latest.csv` (written by lb_poller)
to lb_registry version IDs based on Q-test markers in the description. When a
publicScore appears for a known pending entry, update registry yaml +
append a markdown summary line.

Idempotent: only modifies entries with `outcome: pending`.
"""
from __future__ import annotations
import csv
import re
import time
from io import StringIO
from pathlib import Path

from ._common import ANCHOR_LB, LB_NOISE_BAND, LOG_DIR, REGISTRY_YAML

SUB_CSV = LOG_DIR / "submissions_latest.csv"
SUMMARY_MD = LOG_DIR / "lb_results_summary.md"
INTERVAL_S = 300  # 5min

# Map description-prefix marker → registry version id.
# Re-pushed after apply_state bug fix (2026-05-08 01:55), so v23-v25 ids are
# stale; the actual Kaggle versions are v26/v27/v28; exp175 will push v29.
DESC_TO_ID: dict[str, str] = {
    "Q1: drop all Insecta": "v26",
    "Q2: drop all Amphibia": "v27",
    "Q3: drop rare classes": "v28",
    "exp175 5-fold seed=42": "v30",
    "exp176 5-fold seed=42": "v32",
}


def parse_submissions(csv_text: str) -> list[dict]:
    return list(csv.DictReader(StringIO(csv_text)))


def find_id_for_desc(desc: str) -> str | None:
    for marker, vid in DESC_TO_ID.items():
        if marker in desc:
            return vid
    return None


def update_registry(version_id: str, lb_score: float, raw_desc: str) -> bool:
    """Update lb_registry yaml entry. Returns True if changed, False if not found / already filled."""
    text = REGISTRY_YAML.read_text()
    pattern = (
        rf"(  - id: {version_id}\s*\n"
        rf"(?:    .*\n)*?"
        rf"    expected_lb: \"[^\"]+\"\s*\n"
        rf"    outcome: pending)"
    )
    match = re.search(pattern, text)
    if not match:
        return False

    block = match.group(1)
    delta = lb_score - ANCHOR_LB
    delta_str = f"{delta:+.3f}"
    if abs(delta) <= LB_NOISE_BAND:
        outcome = "matched_anchor"
    elif delta > 0:
        outcome = "improvement"
    else:
        outcome = "regression"

    new_block = block.replace(
        "    outcome: pending",
        (
            f"    lb: {lb_score}\n"
            f"    delta: {delta_str}\n"
            f"    outcome: {outcome}\n"
            f"    auto_updated: \"{time.strftime('%Y-%m-%d %H:%M:%S')}\""
        ),
    )
    new_block = re.sub(r"    lb: pending\b", "", new_block)
    REGISTRY_YAML.write_text(text.replace(block, new_block))
    return True


def append_summary(version_id: str, lb_score: float, desc: str) -> None:
    delta = lb_score - ANCHOR_LB
    line = (
        f"- **{version_id}** ({time.strftime('%Y-%m-%d %H:%M')}): "
        f"LB {lb_score:.4f} (Δ {delta:+.4f}) — {desc[:100]}\n"
    )
    SUMMARY_MD.parent.mkdir(exist_ok=True, parents=True)
    if not SUMMARY_MD.exists():
        SUMMARY_MD.write_text("# LB results summary\n\n")
    with SUMMARY_MD.open("a") as f:
        f.write(line)


def process_once() -> int:
    if not SUB_CSV.exists():
        return 0
    text = SUB_CSV.read_text()
    rows = parse_submissions(text)
    today = time.strftime("%Y-%m-%d")
    yesterday = time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400))

    updated = 0
    for r in rows:
        date = r.get("date", "")
        if not (date.startswith(today) or date.startswith(yesterday)):
            continue
        score = r.get("publicScore", "").strip()
        if not score:
            continue
        try:
            lb = float(score)
        except ValueError:
            continue

        desc = r.get("description", "")
        vid = find_id_for_desc(desc)
        if not vid:
            continue

        if update_registry(vid, lb, desc):
            append_summary(vid, lb, desc)
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] updated {vid} → LB {lb:.4f}", flush=True)
            updated += 1
    return updated


def main():
    print(f"=== lb_processor ({INTERVAL_S//60}min interval) ===", flush=True)
    while True:
        try:
            n = process_once()
            if n == 0:
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] no new updates", flush=True)
        except Exception as e:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] err: {e}", flush=True)
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
