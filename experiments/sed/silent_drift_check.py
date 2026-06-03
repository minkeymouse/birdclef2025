#!/usr/bin/env python3
"""silent_drift_check.py — surface ALL deviations from Tucker spec across our codebase.

Usage:
  uv run python -m experiments.sed.silent_drift_check
"""
from __future__ import annotations
import re
from pathlib import Path
from . import config as cfg_mod
from .tucker_spec import TUCKER

ROOT = Path(__file__).resolve().parents[2]
# Legacy SED scripts live in the pre-sed-refactor archive (since 2026-05-08).
# Drift check still scans them so historical context is preserved.
DP = ROOT / "experiments" / "_archive_2026_pre_sed_refactor"


def _strip_docstrings(text: str) -> str:
    """Remove triple-quoted strings AND line comments to avoid false matches."""
    # Strip ''' or """ blocks
    text = re.sub(r'"""[\s\S]*?"""', "", text)
    text = re.sub(r"'''[\s\S]*?'''", "", text)
    # Strip line comments (# ... to end of line)
    out = []
    for line in text.split("\n"):
        if "#" in line:
            # ignore # inside string literals (heuristic)
            in_str = False
            quote = None
            cut = -1
            for i, ch in enumerate(line):
                if ch in ("'", '"') and (i == 0 or line[i-1] != "\\"):
                    if not in_str:
                        in_str = True; quote = ch
                    elif ch == quote:
                        in_str = False; quote = None
                elif ch == "#" and not in_str:
                    cut = i; break
            if cut >= 0:
                line = line[:cut]
        out.append(line)
    return "\n".join(out)


def _extract_assignment(text: str, key: str) -> str | None:
    """Return value of `key = <value>` from CODE only (post-docstring strip).
    Handles single-line assignments. Multi-statement lines (`A = 1; B = 2`)
    are split first and the cleanest value is returned.
    """
    # Module-level constant: ^KEY = value
    m = re.search(rf"^{re.escape(key)}\s*=\s*([^\n]+)", text, re.MULTILINE)
    if m:
        line = m.group(1).strip()
        # strip trailing comment
        if "#" in line:
            line = line.split("#", 1)[0].rstrip()
        # if multi-statement (semicolon-separated), take first
        if ";" in line:
            line = line.split(";")[0].strip()
        return line.rstrip(";").strip()
    # kwarg pattern (drop_rate=0.1 inside timm.create_model(...))
    m = re.search(rf"\b{re.escape(key)}\s*=\s*([^,\s)]+)", text)
    return m.group(1).strip() if m else None


SCAN_KEYS = [
    "drop_rate", "drop_path_rate",
    "EPOCHS", "BATCH", "LR", "WARMUP", "WD",
    "MIN_SAMPLE",
    "MIXUP_PROB", "MIXUP_ALPHA", "MIXUP_HARD",
    "USE_FOCAL_FOCAL_MIXUP", "USE_FOCAL_SC_MIXUP",
    "N_MELS", "N_FFT", "HOP", "FMIN", "FMAX",
    "EVAL_SS_N_FILES", "USE_AMP", "BG_MIX_P",
]

# Restricted to SED main training scripts (avoid old experiments, helpers)
TARGET_SCRIPTS = [
    "exp169_distilled_sed.py",
    "exp169v2_distilled_sed.py",
    "exp170_distilled_sed.py",
    "exp171_tucker_exact.py",
    "exp172_pseudo_iter1.py",
    "exp173_ds_pseudo_iter2.py",
    "exp174a_finetune_cycle1.py",
    "exp174b_finetune_cycle2.py",
    "exp174_b1_backbone.py",
    "exp175_tucker_actually_exact.py",
]


def scan_legacy_scripts() -> dict[str, dict]:
    """Read each old experiment script and extract values for known params.
    Returns {script_name: {param: value, ...}}.

    Skips docstrings to avoid matching prose like `drop_rate=0.1 (Tucker uses 0)`.
    """
    results = {}
    for fname in TARGET_SCRIPTS:
        fp = DP / fname
        if not fp.exists():
            continue
        raw = fp.read_text()
        text = _strip_docstrings(raw)
        info = {}
        for k in SCAN_KEYS:
            info[k] = _extract_assignment(text, k)
        info["ATT_CLA_INIT_xavier"] = (
            "xavier_uniform_" in text and ("self.att" in text or "self.cla" in text)
        )
        results[fname] = info
    return results


CANONICAL = {
    "drop_rate":              "(absent)",  # Tucker doesn't pass this kwarg
    "drop_path_rate":         "0.1",
    "EPOCHS":                 "25",
    "BATCH":                  "64",
    "LR":                     "5e-4",
    "WARMUP":                 "2",
    "WD":                     "1e-4",
    "MIN_SAMPLE":             "20",
    "MIXUP_PROB":             "0.5",
    "MIXUP_ALPHA":            "0.4",
    "MIXUP_HARD":             "True",
    "USE_FOCAL_FOCAL_MIXUP":  "True",
    "USE_FOCAL_SC_MIXUP":     "True",
    "N_MELS":                 "256",
    "N_FFT":                  "2048",
    "HOP":                    "512",
    "FMIN":                   "20",
    "FMAX":                   "16000",
    "USE_AMP":                "True",
    "ATT_CLA_INIT_xavier":    "True",
}


def diff_vs_tucker(info: dict) -> list[tuple[str, str, str]]:
    """Compare extracted info to Tucker spec."""
    devs = []
    for k, tucker_v in CANONICAL.items():
        ours_v = info.get(k)
        if k == "ATT_CLA_INIT_xavier":
            if ours_v is not True:
                devs.append((k, "True", "False (timm default init)"))
            continue
        if ours_v is None:
            if tucker_v == "(absent)":
                continue  # both absent → ok
            devs.append((k, tucker_v, "(absent)"))
            continue
        # special: drop_rate present = bug (Tucker doesn't have it)
        if k == "drop_rate" and tucker_v == "(absent)":
            devs.append(("drop_rate", "(absent)", str(ours_v)))
            continue
        if str(ours_v).strip() != str(tucker_v).strip():
            devs.append((k, tucker_v, str(ours_v)))
    return devs


def main():
    info_by_script = scan_legacy_scripts()
    print("="*80)
    print("SILENT DRIFT CHECK — every legacy SED script vs Tucker canonical spec")
    print("="*80)
    for script, info in info_by_script.items():
        devs = diff_vs_tucker(info)
        if not devs:
            print(f"\n  ✅ {script}: NO deviations from Tucker spec")
            continue
        print(f"\n  ⚠️  {script}: {len(devs)} deviation(s)")
        for k, tucker, ours in devs:
            print(f"     {k:32s}  tucker={tucker:24s}  ours={ours}")

    # Also scan SedConfig pre-canned configs
    print("\n" + "="*80)
    print("SedConfig pre-canned diffs")
    print("="*80)
    for fn_name in ("tucker_canonical", "exp169_legacy", "exp171_misnamed_tucker",
                    "exp175_tucker_actually", "exp176_per_fold_ss",
                    "exp177_b1_backbone", "exp178_softauc", "exp180_pseudo_v10",
                    "exp183_noisy_student_iter1"):
        cfg = getattr(cfg_mod, fn_name)()
        d = cfg.diff_from_tucker()
        print(f"\n  {fn_name}:")
        if not d:
            print("    ✅ identical to Tucker canonical")
        else:
            for k, v in d.items():
                print(f"    {k:32s}  tucker={v['tucker']!r}  ours={v['ours']!r}")


if __name__ == "__main__":
    main()
