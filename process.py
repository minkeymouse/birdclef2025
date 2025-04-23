#!/usr/bin/env python
"""
process.py – BirdCLEF 2025 preprocessing pipeline
=================================================
Cleans raw *train_audio* / *train_soundscape* audio, applies Voice-Activity
Detection (VAD), deduplicates, builds **10-second mel-spectrogram chunks** with
*soft labels* and sample weighting, then saves:

* `mels/<split>/.../*.npy` – normalized mel spectrogram arrays
* `labels/<split>/.../*.label.npy` – soft-label vectors
* `<split>_metadata.csv` – one row per chunk

Idempotent: safe to rerun after adding/updating raw data or CFG.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import timm

from configure import CFG
from data_utils import (
    compute_mel,
    compute_noise_metric,
    load_audio,
    load_vad,
    remove_speech,
    seed_everything,
    segment_audio,
    trim_silence,
)

#──────────────────────────────────────────────────────────────────────────────
# Global constants driven by CFG
#──────────────────────────────────────────────────────────────────────────────
LABEL_W_PRIMARY       = getattr(CFG, "LABEL_WEIGHT_PRIMARY", 0.95)
LABEL_W_BENCH         = getattr(CFG, "LABEL_WEIGHT_BENCH", 0.05)
RARE_COUNT_THRESHOLD  = getattr(CFG, "RARE_COUNT_THRESHOLD", 20)
PSEUDO_WEIGHT         = getattr(CFG, "PSEUDO_WEIGHT", 0.5)

#──────────────────────────────────────────────────────────────────────────────
# Class discovery & benchmark loader
#──────────────────────────────────────────────────────────────────────────────

def _discover_classes() -> List[str]:
    if CFG.CLASSES:
        return list(CFG.CLASSES)
    if CFG.TAXONOMY_CSV.exists():
        df = pd.read_csv(CFG.TAXONOMY_CSV)
        if "primary_label" in df:
            return sorted(df["primary_label"].unique())
    if CFG.TRAIN_CSV.exists():
        df = pd.read_csv(CFG.TRAIN_CSV)
        if "primary_label" in df:
            return sorted(df["primary_label"].unique())
    raise RuntimeError("Cannot infer species list; set CFG.CLASSES explicitly")

ALL_CLASSES = _discover_classes()
CLASS2IDX   = {s: i for i, s in enumerate(ALL_CLASSES)}

class BenchmarkModel(torch.nn.Module):
    """Load single EfficientNet-B0 for smoothing/validation"""
    def __init__(self, num_classes:int):
        super().__init__()
        self.net = timm.create_model(
            "efficientnet_b0", pretrained=False, in_chans=1, num_classes=num_classes
        )
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_benchmark() -> Optional[torch.nn.Module]:
    path = CFG.BENCHMARK_MODEL
    if not path:
        return None
    bp = Path(path)
    if not bp.exists():
        logging.warning("Benchmark model not found at %s", bp)
        return None
    ck = torch.load(bp, map_location="gpu")
    state = ck.get("model_state_dict", ck)
    m = BenchmarkModel(len(ALL_CLASSES))
    m.load_state_dict(state)
    m.eval()
    return m

#──────────────────────────────────────────────────────────────────────────────
# Utility functions
#──────────────────────────────────────────────────────────────────────────────

def _md5(fp:Path)->str:
    h=hashlib.md5()
    with fp.open("rb") as f:
        for chunk in iter(lambda:f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _deduplicate(paths:Sequence[Path])->List[Path]:
    seen, unique = set(), []
    for p in paths:
        try:
            sig = _md5(p)
        except:
            unique.append(p)
            continue
        if sig not in seen:
            seen.add(sig)
            unique.append(p)
    return unique


def _np_save(fp:Path, arr:np.ndarray)->None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    np.save(fp, arr.astype(np.float32), allow_pickle=False)

#──────────────────────────────────────────────────────────────────────────────
# Soft-label builder
#──────────────────────────────────────────────────────────────────────────────

def _secondary_list(raw)->List[str]:
    if isinstance(raw, str) and raw:
        return [s for s in raw.split(';') if s]
    return []


def build_soft_label(
    primary:str,
    secondaries:List[str],
    bench_model:Optional[torch.nn.Module]=None,
    wav:Optional[np.ndarray]=None,
)->Dict[str,float]:
    label=defaultdict(float)
    rem=1.0 - LABEL_W_BENCH
    if secondaries:
        share=(rem - LABEL_W_PRIMARY)/len(secondaries)
        for s in secondaries:
            label[s]+=share
        label[primary]+=LABEL_W_PRIMARY
    else:
        label[primary]+=rem

    if bench_model and wav is not None:
        m=compute_mel(wav)
        t=torch.from_numpy(m).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits=bench_model(t)
            probs=torch.sigmoid(logits)[0].cpu().numpy()
        for i,p in enumerate(probs):
            if p>0:
                label[ALL_CLASSES[i]]+=LABEL_W_BENCH*p

    total=sum(label.values()) or 1.0
    return {k:v/total for k,v in label.items()}

#──────────────────────────────────────────────────────────────────────────────
# Recording & soundscape processing
#──────────────────────────────────────────────────────────────────────────────

def _process_recordings():
    log=logging.getLogger()
    log.info("✨ Processing train_audio recordings...")

    df=pd.read_csv(CFG.TRAIN_CSV)
    if CFG.MIN_RATING>0 and 'rating' in df:
        df=df[df['rating']>=CFG.MIN_RATING]

    recs=[]
    for r in df.itertuples(index=False):
        fp=CFG.TRAIN_AUDIO_DIR/ r.filename
        if not fp.exists():
            log.warning("Missing %s",fp)
            continue
        d=r._asdict() if hasattr(r,'_asdict') else r._asdict()
        d['filepath']=fp
        recs.append(d)
    df=pd.DataFrame(recs)
    if df.empty:
        log.warning("No files to process."); return

    df=df[df['filepath'].isin(_deduplicate(df['filepath'].tolist()))].reset_index(drop=True)
    df['noise']= [compute_noise_metric(load_audio(fp)) for fp in df['filepath']]
    ratio=CFG.FOLD0_RATIO
    if 0<ratio<1:
        th=np.quantile(df['noise'],ratio)
        df=df[df['noise']<=th].reset_index(drop=True)
        log.info("Using %d clean recordings",len(df))

    bench=_load_benchmark()
    if bench: log.info("Benchmark loaded.")
    vad,ts=load_vad()

    mel_dir=CFG.PROCESSED_DIR/'mels'/'train'
    lbl_dir=CFG.PROCESSED_DIR/'labels'/'train'

    rows=[]
    for r in df.itertuples(index=False):
        y=load_audio(r.filepath)
        y=trim_silence(y)
        y=remove_speech(y,vad,ts)
        if np.sqrt((y**2).mean())<CFG.RMS_THRESHOLD: continue

        secs=list(segment_audio(y))
        secs2=_secondary_list(getattr(r,'secondary_labels',''))
        prim=r.primary_label
        if bench:
            mfull=compute_mel(y)
            t1=torch.from_numpy(mfull).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                pfull=torch.sigmoid(bench(t1))[0].cpu().numpy()
            top=ALL_CLASSES[int(pfull.argmax())]
            if top!=prim:
                if top in secs2: prim=top
                else: continue

        for st,chunk in secs:
            L=CFG.TRAIN_CHUNK_SEC*CFG.SAMPLE_RATE
            if len(chunk)<L: chunk=np.pad(chunk,(0,L-len(chunk)),'wrap')
            mel=compute_mel(chunk)
            soft=build_soft_label(prim,secs2,bench,chunk)

            stem=f"{r.filepath.stem}_{int(st)}s"
            mp=mel_dir/prim/f"{stem}.npy"; lp=lbl_dir/prim/f"{stem}.label.npy"
            _np_save(mp,mel); _np_save(lp,np.array([soft.get(c,0) for c in ALL_CLASSES],dtype=np.float32))

            rows.append({
                'mel_path':str(mp.relative_to(CFG.PROCESSED_DIR)),
                'label_path':str(lp.relative_to(CFG.PROCESSED_DIR)),
                'label_json':json.dumps(soft,separators=(',',':')),
                'duration':CFG.TRAIN_CHUNK_SEC,
                'noise_score':getattr(r,'noise',0),
                'weight':1.0,
            })

    pd.DataFrame(rows).to_csv(CFG.PROCESSED_DIR/'train_metadata.csv',index=False)
    log.info("Wrote %d train chunks",len(rows))


def _process_soundscapes():
    log=logging.getLogger()
    if not CFG.TRAIN_SOUNDSCAPE_DIR.exists(): log.info("No soundscapes"); return
    bench=_load_benchmark()
    if not bench:
        raise RuntimeError("Benchmark model not found; cannot process soundscapes")
    vad,ts=load_vad()

    mel_dir=CFG.PROCESSED_DIR/'mels'/'soundscape'
    lbl_dir=CFG.PROCESSED_DIR/'labels'/'soundscape'

    rows=[]
    for fp in sorted(CFG.TRAIN_SOUNDSCAPE_DIR.glob('*.ogg')):
        y=load_audio(fp); y=trim_silence(y); y=remove_speech(y,vad,ts)
        for st,seg in segment_audio(y):
            L=CFG.TRAIN_CHUNK_SEC*CFG.SAMPLE_RATE
            if len(seg)<L: seg=np.pad(seg,(0,L-len(seg)),'wrap')
            mel=compute_mel(seg)
            if bench:
                with torch.no_grad():
                    p=torch.sigmoid(bench(torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)))[0].cpu().numpy()
            else:
                p=np.ones(len(ALL_CLASSES),dtype=np.float32)
            if float(p.max())<CFG.PSEUDO_THRESHOLD: continue
            stem=f"{fp.stem}_{int(st)}s"
            mp=mel_dir/f"{stem}.npy"; lp=lbl_dir/f"{stem}.label.npy"
            _np_save(mp,mel); _np_save(lp,p)
            soft={ALL_CLASSES[i]:float(v) for i,v in enumerate(p) if v>0}
            rows.append({
                'mel_path':str(mp.relative_to(CFG.PROCESSED_DIR)),
                'label_path':str(lp.relative_to(CFG.PROCESSED_DIR)),
                'label_json':json.dumps(soft,separators=(',',':')),
                'duration':CFG.TRAIN_CHUNK_SEC,
                'noise_score':compute_noise_metric(seg),
                'weight':PSEUDO_WEIGHT,
            })
    if rows:
        pd.DataFrame(rows).to_csv(CFG.PROCESSED_DIR/'soundscape_metadata.csv',index=False)
        log.info("Wrote %d pseudo-labelled chunks",len(rows))


def _apply_rare_weighting():
    files=[CFG.PROCESSED_DIR/'train_metadata.csv', CFG.PROCESSED_DIR/'soundscape_metadata.csv']
    dfs=[pd.read_csv(f) for f in files if f.exists()]
    if not dfs: return
    allm=pd.concat(dfs,ignore_index=True)
    cnt=Counter(max(json.loads(j), key=json.loads(j).get) for j in allm['label_json'])
    rares={s for s,c in cnt.items() if c<RARE_COUNT_THRESHOLD}
    for f in files:
        if not f.exists(): continue
        df=pd.read_csv(f)
        df['weight']=[CFG.RARE_WEIGHT if max(json.loads(j),key=json.loads(j).get) in rares else 1.0 for j in df['label_json']]
        df.to_csv(f,index=False)
    logging.getLogger().info("Applied rare weighting to %d species", len(rares))


def main():
    p=argparse.ArgumentParser(description="Preprocess BirdCLEF 2025 audio")
    p.add_argument('--verbose',action='store_true')
    args=p.parse_args()
    lvl=logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=lvl, format="%(asctime)s [%(levelname)s] %(message)s")
    seed_everything(CFG.SEED)
    CFG.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    _process_recordings(); _process_soundscapes(); _apply_rare_weighting()
    logging.info("✅ Done – data in %s", CFG.PROCESSED_DIR)

if __name__=='__main__': main()
