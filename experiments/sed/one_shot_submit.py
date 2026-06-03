"""one_shot_submit — generic submitter for a specific Kaggle kernel version.

Replaces the ad-hoc per-version _vXX_submit.py scripts. Polls submit until
Kaggle has finished re-running the version, then submits with the given message.

Usage:
  KAGGLE_API_TOKEN=KGAT_... uv run python -m experiments.sed.one_shot_submit \\
    --version 36 --message "..."
"""
from __future__ import annotations
import argparse
import time

from ._common import try_submit


def submit_loop(version: int, message: str, *, max_retries: int = 60,
                interval_s: int = 120) -> bool:
    fails = 0
    while True:
        ok, out = try_submit(version, message)
        ts = time.strftime("%H:%M:%S")
        if ok:
            print(f"[{ts}] SUBMITTED v{version} ✓\n{out}", flush=True)
            return True
        fails += 1
        if fails > max_retries:
            print(f"[{ts}] giving up on v{version} after {fails} attempts", flush=True)
            return False
        print(f"[{ts}] v{version} {out[:80]} (#{fails})", flush=True)
        time.sleep(interval_s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, required=True,
                        help="Kaggle kernel version to submit")
    parser.add_argument("--message", type=str, required=True,
                        help="Submission description")
    args = parser.parse_args()

    submit_loop(args.version, args.message)


if __name__ == "__main__":
    main()
