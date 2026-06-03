"""notebook_state.py — single source of truth for notebook configuration.

Replaces 6+ scattered patch scripts (exp170_add_konbu_head, exp170_phase1_robin_hood,
exp170_slot{1,2,3}_*, exp171_deploy, etc) which were stateful and easy to lose
track of. Each previous patch modified or appended cells; reverts were
inconsistent.

This module declares notebook configuration as DATA, applies it idempotently:
no-op if already in target state, deterministic if not.

Usage:
  from experiments.sed.notebook_state import NotebookState, apply_state, current_state
  ns = NotebookState(
      sed_dataset="ultimatumgame/exp175-distilled-sed",
      sed_finder_dir_token="exp175",
      ulyanov_blend=False,    # use Mattia 0.941 blend
      konbu_head=False,       # no konbu head
      konbu_head_weight=0.05,
  )
  apply_state(ns, "/path/to/notebook.ipynb")
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

# Stable cell content templates — generate from these instead of in-place patching.
TUCKER_FIND_BLOCK_TEMPLATE = """def find_{token}_dir():
    # Match both 1-digit (sed_fold0.onnx) and zero-padded (sed_fold00.onnx)
    hits = sorted(Path("/kaggle/input").rglob("sed_fold*.onnx"))
    # Filter by Kaggle dataset slug (Kaggle mounts at /kaggle/input/<slug>/)
    candidates = [p.parent for p in hits if "{slug}" in str(p)]
    return candidates[0] if candidates else None

{token}_dir = find_{token}_dir()
if {token}_dir is None:
    raise FileNotFoundError("Attach ultimatumgame/{slug} to this notebook.")

_all_sed_paths = sorted(
    {token}_dir.glob("sed_fold*.onnx"),
    key=lambda p: int(re.search(r"sed_fold(\\d+)", p.name).group(1))
)
SED_FOLD_FILTER = {fold_filter}
if SED_FOLD_FILTER is not None:
    sed_fold_paths = [_all_sed_paths[i] for i in SED_FOLD_FILTER if i < len(_all_sed_paths)]
else:
    sed_fold_paths = _all_sed_paths

sed_dir = {token}_dir
sed_sessions = [make_sed_session(p) for p in sed_fold_paths]

print(f"SED dir: {{sed_dir}}  ({token})  [{{len(sed_fold_paths)}}/{{len(_all_sed_paths)}} folds, filter={{SED_FOLD_FILTER}}]")
print(f"SED folds loaded: {{[p.name for p in sed_fold_paths]}}")"""


@dataclass(frozen=True)
class NotebookState:
    """Declarative notebook configuration for one LB submission attempt."""
    sed_dataset: str = "ultimatumgame/exp169-distilled-sed"
    sed_finder_dir_token: str = "exp169"
    ulyanov_blend: bool = False
    konbu_head: bool = False
    konbu_head_weight: float = 0.05  # Ulyanov uses 0.05 (not v6/v7's 0.15)
    sonotype_mirror: bool = False    # Path A: needless090 0.934 Insecta sonotype groups
    rare_suppression: bool = False   # Path G: Amphibia/Mammalia/Reptilia low-conf × 0.9 (needless090 0.946 trick)
    sed_fold_filter: tuple | None = None  # subset of fold indices within sed_dataset (None = all)


def apply_state(state: NotebookState, nb_path: Path) -> None:
    """Idempotent: replace SED-loader and blend cells based on state.
    Always reads notebook fresh, computes target cell content, replaces.
    """
    nb_path = Path(nb_path)
    nb = json.loads(nb_path.read_text())

    # 1. SED loader cell (find_<token>_dir + session loading)
    fold_filter_repr = repr(list(state.sed_fold_filter)) if state.sed_fold_filter is not None else "None"
    target_loader = TUCKER_FIND_BLOCK_TEMPLATE.format(
        token=state.sed_finder_dir_token,
        slug=state.sed_dataset.split("/")[-1],
        fold_filter=fold_filter_repr,
    )
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if re.search(r"def find_(?:exp\w+|sed)_dir|find_sed_dirs", src):
            # Replace ONLY the second-and-later `def find_*` block (the dataset patch),
            # preserving the helper functions (make_sed_session, audio_to_mel, etc)
            # which are defined as the FIRST `def find_sed_dir` block.
            #
            # v5 anchor structure:
            #   def find_sed_dir()         ← unused stub (preserved)
            #   def make_sed_session(p)    ← helper (must preserve)
            #   def audio_to_mel(...)      ← helper (must preserve)
            #   def file_to_sed_chunks(...)← helper (must preserve)
            #   def sigmoid_sed(...)       ← helper (must preserve)
            #   # Load the 5 SED fold models
            #   # ===== <patch comment block>
            #   def find_exp169_dir()      ← REPLACE FROM HERE
            #   ...
            #   sed_sessions = [...]       ← REPLACE TO HERE
            patch_marker = "# ===== "
            patch_start = src.find(patch_marker)
            if patch_start < 0:
                # Fallback: find the last `def find_` block
                idx = 0
                last_find = -1
                while True:
                    nxt = src.find("def find_", idx)
                    if nxt < 0:
                        break
                    last_find = nxt
                    idx = nxt + 1
                start = last_find
            else:
                start = patch_start

            end = src.find("sed_sessions = [make_sed_session(p) for p in sed_fold_paths]")
            if end < 0:
                end = src.find("sed_sessions = [")
            if start >= 0 and end > start:
                line_end = src.find("\n", end)
                if line_end < 0:
                    line_end = len(src)
                new_src = src[:start] + target_loader + src[line_end:]
                cell["source"] = [new_src]
            break

    # 2. Optionally inject konbu head cell + Ulyanov blend
    # (Implementation skipped here — to be added when needed; today's focus
    # is documenting the abstraction.)

    # 3. Optionally inject sonotype mirror + rare suppression (MirrorRare = needless090 0.946 trick).
    if state.sonotype_mirror or state.rare_suppression:
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            # Mattia + Ulyanov both end with `sub.to_csv(OUT_CSV, index=False)`
            anchor = "sub.to_csv(OUT_CSV, index=False)"
            if anchor not in src:
                continue
            if "MIRROR_PAIRS" in src or "MIRRORRARE_INJECTED" in src:
                # already injected; idempotent no-op
                break
            mirror_part = (
                "MIRROR_PAIRS = (\n"
                '    ("47158son15", "47158son16"),\n'
                '    ("47158son09", "47158son12"),\n'
                '    ("47158son02", "47158son14"),\n'
                '    ("47158son13", "47158son21", "47158son22", "47158son23"),\n'
                ")\n"
                "_l2i = {l: i for i, l in enumerate(cols)}\n"
                "_pred_orig = sub[cols].to_numpy(np.float32).copy()\n"
                "_pred_mirror = _pred_orig.copy()\n"
                "for _g in MIRROR_PAIRS:\n"
                "    _ix = [_l2i[s] for s in _g if s in _l2i]\n"
                "    if len(_ix) >= 2:\n"
                "        _m = _pred_mirror[:, _ix].max(axis=1, keepdims=True)\n"
                "        for _i in _ix:\n"
                "            _pred_mirror[:, _i] = _m.squeeze()\n"
                "sub[cols] = _pred_mirror\n"
                "print(f'Sonotype mirror applied: {len(MIRROR_PAIRS)} groups, "
                "abs delta sum={(np.abs(_pred_mirror - _pred_orig)).sum():.2f}')\n"
            ) if state.sonotype_mirror else ""
            rare_part = (
                "_tax = pd.read_csv(BASE / 'taxonomy.csv').set_index('primary_label')\n"
                "_rare_classes = ['Amphibia', 'Mammalia', 'Reptilia']\n"
                "_rare_n = 0\n"
                "for _ci, _sp in enumerate(cols):\n"
                "    if _sp in _tax.index and _tax.loc[_sp, 'class_name'] in _rare_classes:\n"
                "        _vals = sub[_sp].to_numpy(np.float32)\n"
                "        _thr = _vals.mean() + 0.05\n"
                "        sub[_sp] = np.where(_vals < _thr, _vals * 0.9, _vals)\n"
                "        _rare_n += 1\n"
                "print(f'Rare suppression applied to {_rare_n} species (Amphibia/Mammalia/Reptilia)')\n"
            ) if state.rare_suppression else ""
            tags = []
            if state.sonotype_mirror:
                tags.append("sonotype_mirror")
            if state.rare_suppression:
                tags.append("rare_suppression")
            patch = (
                f"\n# === MirrorRare ({'+'.join(tags)}) — needless090 0.946 trick ===\n"
                "# MIRRORRARE_INJECTED  (idempotency marker)\n"
                + mirror_part + rare_part +
                "sub.to_csv(OUT_CSV, index=False)\n"
            )
            new_src = src.replace(anchor, patch.lstrip(), 1)
            cell["source"] = [new_src]
            break

    nb_path.write_text(json.dumps(nb))


def current_state(nb_path: Path) -> dict:
    """Inspect notebook and report its current configuration as a dict."""
    nb = json.loads(Path(nb_path).read_text())
    out = {"sed_dir_tokens_seen": [], "has_konbu_head": False,
           "has_dual_pc_blend": False, "blend_cell_marker": None}
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        for tok in ("exp169", "exp171", "exp174a", "exp174b", "exp175"):
            if f"find_{tok}_dir" in src or f"{tok}_dir =" in src:
                out["sed_dir_tokens_seen"].append(tok)
        if "head_weights_train_audio.npz" in src:
            out["has_konbu_head"] = True
        if "make_gate_pred" in src or "PC010_WEIGHT" in src:
            out["has_dual_pc_blend"] = True
        if "# Cell 3 — SED-smoothed rank ensemble" in src:
            out["blend_cell_marker"] = "mattia_0941"
        elif "# Cell 3 - dual 0.945 final-layer average" in src:
            out["blend_cell_marker"] = "ulyanov_0945"
    return out
