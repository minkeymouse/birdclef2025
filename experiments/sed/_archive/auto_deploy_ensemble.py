"""auto_deploy_ensemble — wait for chain to finish, deploy ensemble, submit.

Polls every 5min until exp177 fold 4 has best_ckpt.pt + history.json. Then runs
deploy_ensemble.main() and submits the resulting kernel version.
"""
from __future__ import annotations
import json
import os
import subprocess
import time

from ._common import ROOT, get_last_pushed_version, try_submit

POLL_S = 300  # 5min
SUBMIT_POLL_S = 120
MAX_SUBMIT_CYCLES = 60


def chain_done() -> bool:
    """Chain done when exp177 fold 4 history.json shows epoch >= 25."""
    fold_dir = ROOT / "experiments/_data_pipelines/exp177_outputs/seed42/fold4"
    ck = fold_dir / "best_ckpt.pt"
    hist = fold_dir / "history.json"
    if not (ck.exists() and hist.exists()):
        return False
    try:
        h = json.loads(hist.read_text())
        return h.get("history", [{}])[-1].get("epoch", 0) >= 25
    except Exception:
        return False


def wait_for_chain():
    while not chain_done():
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] waiting for exp177 fold 4 to complete ({POLL_S//60}min poll)", flush=True)
        time.sleep(POLL_S)
    print("=== CHAIN COMPLETE ===", flush=True)


def deploy() -> bool:
    print("=== Running deploy_ensemble.main() ===", flush=True)
    proc = subprocess.run(
        ["uv", "run", "python", "-m", "experiments.sed.deploy_ensemble"],
        cwd=ROOT, env=os.environ.copy(), text=True,
    )
    return proc.returncode == 0


def submit_when_ready(version: int):
    msg = (
        "Multi-seed + multi-arch ensemble: 25 ckpts. "
        "exp175 (B0) seed=42,43,44 + exp176 (B0, per-fold SS) seed=42 + "
        "exp177 (B1 backbone) seed=42. Mattia 0.941 blend. "
        "Tests if averaging across recipe-equivalent SEDs catches up to Tucker (0.941)."
    )
    fails = 0
    while True:
        ok, out = try_submit(version, msg)
        ts = time.strftime("%H:%M:%S")
        if ok:
            print(f"[{ts}] SUBMITTED v{version} ✓\n{out[-200:]}", flush=True)
            return
        fails += 1
        if fails > MAX_SUBMIT_CYCLES:
            print(f"[{ts}] giving up on v{version}", flush=True)
            return
        print(f"[{ts}] v{version} {out[:80]} (#{fails})", flush=True)
        time.sleep(SUBMIT_POLL_S)


def main():
    print("=== auto_deploy_ensemble ===", flush=True)
    wait_for_chain()
    if not deploy():
        print("DEPLOY FAILED", flush=True)
        return
    v = get_last_pushed_version()
    if v is None:
        print("ERROR: pushed version not found", flush=True)
        return
    print(f"=== Waiting for Kaggle re-run of v{v} before submit ===", flush=True)
    submit_when_ready(v)


if __name__ == "__main__":
    main()
