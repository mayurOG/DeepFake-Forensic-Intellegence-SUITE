"""Train the optional learned temporal head on cached frame embeddings.

The statistical temporal aggregator works without training; this head
replaces it when you have labelled videos and learns *when and how* frames
disagree over time.

    python -m train.train_temporal --data data/ffpp --encoder <model-id> --out artifacts
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, Sequence, Tuple

import numpy as np

from detector import metrics as MT
from detector.experts.temporal import build_temporal_head
from train.data import Item, group_split, scan_folder

log = logging.getLogger("deepfake.train")


def fit_head(X: np.ndarray, y: np.ndarray, epochs: int = 80, lr: float = 1e-3,
             hidden: int = 64, val_fraction: float = 0.2, seed: int = 0):
    """X: (N, T, D) per-video frame-embedding sequences, y: (N,).
    Returns (head, metrics)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_fraction)) if n >= 5 else 0
    va, tr = idx[:n_val], idx[n_val:]
    Xt = torch.tensor(X[tr], dtype=torch.float32)
    yt = torch.tensor(y[tr], dtype=torch.float32)
    head = build_temporal_head(X.shape[2], hidden)
    n_fake = max(1.0, float(yt.sum()))
    pos_w = torch.tensor([(len(yt) - n_fake) / n_fake])
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(perm), 16):
            b = perm[i:i + 16]
            loss = loss_fn(head(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    metrics: Dict[str, float] = {}
    if n_val:
        with torch.no_grad():
            z = head(torch.tensor(X[va], dtype=torch.float32)).numpy()
        metrics["val_auc"] = MT.auc(y[va], z)
    return head, metrics


def extract_sequences(items: Sequence[Item], encoder, T: int = 12, use_faces: bool = True):
    from detector import models as M
    from train.data import load_images
    from train.features import face_crop

    det = M.get_face_detector() if use_faces else None
    seqs, ys, gs = [], [], []
    for it in items:
        if it.media_type != "video":
            continue
        crops = [face_crop(im, det, use_faces) for im in load_images(it, T)]
        if len(crops) < 3:
            continue
        e = encoder.embed(crops)
        if len(e) < T:  # pad by repeating the last embedding
            e = np.concatenate([e, np.repeat(e[-1:], T - len(e), axis=0)])
        seqs.append(e[:T])
        ys.append(it.label)
        gs.append(it.group)
    return np.stack(seqs), np.asarray(ys), np.asarray(gs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--no-faces", action="store_true")
    ap.add_argument("--max-items", type=int, default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    from train.features import Encoder

    items = [i for i in scan_folder(a.data, media="video", max_items=a.max_items)]
    X, y, _ = extract_sequences(items, Encoder(a.encoder), a.frames, not a.no_faces)
    head, metrics = fit_head(X, y, a.epochs)
    import torch

    os.makedirs(a.out, exist_ok=True)
    torch.save({"state": head.state_dict(), "dim": int(X.shape[2]), "hidden": 64,
                "encoder": a.encoder}, os.path.join(a.out, "temporal_head.pt"))
    print(metrics, f"saved to {a.out}/temporal_head.pt")


if __name__ == "__main__":
    main()
