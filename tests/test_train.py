import json
import os

import numpy as np
import pytest

from detector import models as M
from detector.calibration import CalibrationBundle
from detector.pipeline import Pipeline
from tests.conftest import stub_image_fn
from train import evaluate, fit_calibration, train_probe, features, finetune_ln, train_temporal
from train.data import scan_folder, group_split


def test_scan_and_group_split(dataset):
    items = scan_folder(dataset, "image")
    assert len(items) == 48 and {i.label for i in items} == {0, 1}
    tr, va = group_split(items, 0.25)
    assert not ({i.group for i in tr} & {i.group for i in va})  # no leakage


def test_features_probe_and_probe_backend(dataset, tiny_encoder, tmp_path, settings):
    out = str(tmp_path / "f.npz")
    features.main(["--data", dataset, "--encoder", tiny_encoder, "--out", out, "--no-faces"])
    d = np.load(out, allow_pickle=True)
    assert d["emb"].shape == (48, 32)
    probe = str(tmp_path / "probe.npz")
    train_probe.main(["--train", out, "--out", probe])
    assert os.path.exists(probe)
    be = M.ProbeBackend(tiny_encoder, probe)
    from tests.conftest import make_img
    z, emb = be.predict([make_img(50, 0, -50, size=96), make_img(-50, 0, 50, size=96)])
    assert np.asarray(z).shape == (2,) and np.asarray(emb).shape[-1] == 32


def test_hf_classifier_backend_tiny(tmp_path):
    from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor
    cfg = ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                    intermediate_size=64, image_size=32, patch_size=16,
                    id2label={0: "Realism", 1: "Deepfake"}, label2id={"Realism": 0, "Deepfake": 1})
    d = str(tmp_path / "clf")
    ViTForImageClassification(cfg).save_pretrained(d)
    ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(d)
    be = M.HFClassifierBackend(d)
    from tests.conftest import make_img
    z, emb = be.predict([make_img(0, 0, 0, size=64)])
    assert np.isfinite(np.asarray(z)).all()


def test_calibrate_then_evaluate(dataset, pipe, settings, tmp_path, monkeypatch):
    items = scan_folder(dataset, "image")
    rows = fit_calibration.collect_scores(pipe, items, "image")
    assert len(rows) == 48
    rep, bundle = fit_calibration.fit_from_rows(rows, "image", 0.1, out_dir=settings.artifact_dir)
    bundle.save(os.path.join(settings.artifact_dir, "calibration.json"))
    assert rep["final"]["auc"] > 0.95 and rep["ood_fitted"]
    assert os.path.exists(os.path.join(settings.artifact_dir, "calibration.json"))
    pipe.reload()
    assert pipe.status()  # loads calibration
    res = evaluate.evaluate_items(pipe, items, "image", "balanced")
    assert res["cascade"]["auc"] > 0.95 and res["decisions"]["false_alarm_rate"] <= 0.1
    json.dumps(res)


def test_finetune_ln_smoke(dataset, tiny_encoder, tmp_path):
    out = str(tmp_path / "ft")
    finetune_ln.main(["--data", dataset, "--encoder", tiny_encoder, "--out", out,
                      "--epochs", "1", "--batch-size", "8", "--no-faces", "--max-items", "16"])
    assert os.path.exists(os.path.join(out, "probe.npz"))


def test_temporal_head_fit():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 80)
    X = rng.normal(0, 1, (80, 12, 8)) + y[:, None, None] * 0.8
    head = train_temporal.fit_head(X, y, epochs=30)
    assert head is not None
