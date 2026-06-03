"""lb_poller — periodically dump submissions CSV.

Every 10min, write submissions CSV to a known path. lb_processor reads from
this file (plus the registry yaml) to detect newly-scored submissions and
populate registry entries.
"""
from __future__ import annotations
import subprocess
import time

from ._common import COMP_SLUG, LOG_DIR, kaggle_env

OUT_CSV = LOG_DIR / "submissions_latest.csv"
INTERVAL_S = 600  # 10min


def fetch_submissions() -> str:
    proc = subprocess.run(
        ["uv", "run", "kaggle", "competitions", "submissions",
         "-c", COMP_SLUG, "--csv"],
        env=kaggle_env(), capture_output=True, text=True, timeout=60,
    )
    return proc.stdout


def main():
    print(f"=== lb_poller (interval {INTERVAL_S}s) ===", flush=True)
    while True:
        try:
            csv_text = fetch_submissions()
            if csv_text:
                OUT_CSV.write_text(csv_text)
                lines = csv_text.strip().split("\n")
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] wrote {len(lines)} submissions to {OUT_CSV.name}", flush=True)
                for line in lines[1:4]:
                    parts = line.split(",", 5)
                    if len(parts) >= 5:
                        date, score = parts[1][:19], parts[4]
                        print(f"    {date}  score={score}", flush=True)
        except Exception as e:
            print(f"  err: {e}", flush=True)
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
