"""Robust LB score poller: waits until the most-recent submission is scored, then exits.
Skips the kaggle CLI 'Warning' line that pollutes stdout."""
import subprocess, time, csv, io, sys

COMP = "birdclef-2026"
while True:
    p = subprocess.run(
        ["uv", "run", "kaggle", "competitions", "submissions", "-c", COMP, "--csv"],
        capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines() if not l.lstrip().startswith("Warning")]
    rows = [r for r in csv.reader(io.StringIO("\n".join(lines))) if r and r[0] == "submission.csv"]
    if rows:
        top = rows[0]                      # most recent submission
        status, public = top[3], top[4]
        print(time.strftime("%H:%M:%S"), status, "public=" + repr(public), flush=True)
        if public.strip() not in ("", None) or "COMPLETE" in status:
            print("SCORED", "public=" + repr(public)); break
        if "ERROR" in status:
            print("ERRORED"); break
    else:
        print(time.strftime("%H:%M:%S"), "no data row yet", flush=True)
    time.sleep(120)
