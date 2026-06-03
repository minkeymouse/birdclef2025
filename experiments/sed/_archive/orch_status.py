"""orch_status — single-shot snapshot of all autonomous orchestration state.

Prints:
  - Background processes still alive (filtered to relevant ones)
  - exp175/exp176 fold completion status
  - Latest log tails for each orchestration component
  - lb_registry pending entries
  - Recent LB submissions (today + yesterday)
"""
from __future__ import annotations
import re
import subprocess
import time
from pathlib import Path

from ._common import LOG_DIR, REGISTRY_YAML, ROOT


PROCESS_PATTERNS = (
    "exp175_tucker", "exp176_chained", "q_test_submit",
    "auto_deploy_exp175", "lb_poller", "lb_processor",
    "experiments.sed",
)


def show_processes() -> None:
    print("=== Background processes ===", flush=True)
    proc = subprocess.run(["ps", "-ef"], capture_output=True, text=True)
    for line in proc.stdout.split("\n"):
        if "grep" in line or "ps -ef" in line:
            continue
        if not any(p in line for p in PROCESS_PATTERNS):
            continue
        fields = line.split()
        if len(fields) >= 7:
            pid = fields[1]
            cmd = " ".join(fields[7:])[:120]
            print(f"  PID {pid}: {cmd}", flush=True)


def show_folds() -> None:
    """Show fold progress per (config, seed) bucket."""
    import json
    print("\n=== Fold status (per config/seed) ===", flush=True)
    targets = [
        ("exp175", 42, "experiments/_data_pipelines/exp175_outputs/seed42"),
        ("exp175", 43, "experiments/_data_pipelines/exp175_outputs/seed43"),
        ("exp175", 44, "experiments/_data_pipelines/exp175_outputs/seed44"),
        ("exp176", 42, "experiments/_data_pipelines/exp176_outputs"),
        ("exp177-B1", 42, "experiments/_data_pipelines/exp177_outputs/seed42"),
    ]
    for label, seed, rel in targets:
        out_dir = ROOT / rel
        if not out_dir.exists():
            print(f"  {label} seed={seed}: not started", flush=True)
            continue
        states = []
        for f in range(5):
            ck = out_dir / f"fold{f}/best_ckpt.pt"
            hist = out_dir / f"fold{f}/history.json"
            if not ck.exists():
                states.append(".")
                continue
            if not hist.exists():
                states.append("·")  # ckpt only, training in progress
                continue
            try:
                data = json.loads(hist.read_text())
                last = data.get("history", [{}])[-1].get("epoch", 0)
                states.append("✓" if last >= 25 else f"e{last}")
            except Exception:
                states.append("?")
        print(f"  {label} seed={seed}: {' '.join(states)}", flush=True)


def show_log_tails() -> None:
    print("\n=== Latest log tails ===", flush=True)
    targets = [
        ("exp175_resume", 3),
        ("exp176_chained", 3),
        ("q_test_submit", 5),
        ("auto_deploy_exp175", 3),
        ("lb_poller", 3),
        ("lb_processor", 3),
    ]
    for prefix, n in targets:
        logs = sorted(LOG_DIR.glob(f"{prefix}*.log"))
        if not logs:
            print(f"  {prefix}: no log", flush=True)
            continue
        latest = logs[-1]
        try:
            lines = latest.read_text().strip().split("\n")
            print(f"  --- {latest.name} (last {min(n, len(lines))} lines) ---", flush=True)
            for line in lines[-n:]:
                print(f"    {line[:200]}", flush=True)
        except Exception as e:
            print(f"    err: {e}", flush=True)


def show_lb_registry_pending() -> None:
    print("\n=== lb_registry pending entries ===", flush=True)
    text = REGISTRY_YAML.read_text()
    # Split into entries; check each for outcome: pending
    entries = re.split(r"(?=^  - id: v\d+)", text, flags=re.MULTILINE)
    for ent in entries:
        m = re.match(r"  - id: (v\d+)\s*\n", ent)
        if not m or "outcome: pending" not in ent:
            continue
        vid = m.group(1)
        hyp_match = re.search(r"hypothesis: (?:\|\s*\n\s*)?([^\n]+)", ent)
        hyp = hyp_match.group(1) if hyp_match else "(no hypothesis)"
        print(f"  {vid}: {hyp[:120]}", flush=True)


def show_recent_submissions() -> None:
    print("\n=== Recent LB submissions (today + yesterday) ===", flush=True)
    csv_file = LOG_DIR / "submissions_latest.csv"
    if not csv_file.exists():
        print("  no submissions file yet", flush=True)
        return
    text = csv_file.read_text()
    today = time.strftime("%Y-%m-%d")
    yesterday = time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400))
    for line in text.split("\n")[1:]:
        if not (today in line or yesterday in line):
            continue
        parts = line.split(",", 5)
        if len(parts) >= 5:
            date = parts[1][:19]
            score = parts[4]
            desc_short = parts[2].lstrip('"')[:60] if len(parts[2]) > 1 else ""
            print(f"    {date}  score={score}  {desc_short}", flush=True)


def main():
    print(f"=== orch_status @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n", flush=True)
    show_processes()
    show_folds()
    show_log_tails()
    show_lb_registry_pending()
    show_recent_submissions()


if __name__ == "__main__":
    main()
