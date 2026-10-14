# 🛡️ Deepfake Forensic Intelligence Suite

Multi-modal (image, video, audio) deepfake analysis built as a **calibrated
evidence cascade**: every verdict is *Real*, *Fake* or *Inconclusive*, comes
with a calibrated probability, an explanation and a cryptographically signed
report.

> **Honest status.** The architecture, training, calibration and evaluation
> tooling are implemented and unit-tested on synthetic data with stub / tiny
> local models (`pytest`, 29 tests). **No accuracy figure is claimed.** Real
> accuracy depends on the backbone you train and the datasets you calibrate
> on. Measure it with `train.evaluate` on data the model has never seen
> (ideally a different dataset / generator than training).

## How it works

```
input → triage (hash, C2PA presence, quality) → Tier-1 fast screen
      → early exit if the conformal set is a single label and not out-of-domain
      → otherwise Tier-2 experts (tamper map, extra backbones, temporal head,
        rPPG, audio-visual sync, landmark geometry)
      → reliability-weighted fusion of log-odds (disagreement shrinks confidence)
      → calibration (Platt) + class-conditional conformal prediction
      → Real | Fake | Inconclusive
      → occlusion heatmap + faithfulness score, re-encoding stability test
      → signed JSON + HTML report → drift canary
```

* **Abstains instead of guessing.** Conformal prediction sets give a coverage
  target (`DEEPFAKE_ALPHA`, default 0.1). Out-of-domain inputs (Mahalanobis
  distance on embeddings) become *Inconclusive*.
* **Operating modes:** `strict` (~1% false alarms), `balanced` (~5%),
  `sensitive` (~10%). Without a fitted calibration the system says so and uses
  a conservative margin around 0.5.
* **Tier-2 heuristics** (rPPG, sync, geometry, tamper) are low-weight,
  explicitly labelled as heuristics, and cannot override a strong neural score.
* **C2PA:** presence of a manifest is reported; its signature is *not* verified.
* **Signed reports:** Ed25519 key auto-created at
  `artifacts/report_signing_key.pem` (keep it private, it is git-ignored).
  Verify with `python -m scripts.verify_report report.json`.

## Install

```bash
git clone https://github.com/mayurOG/DEEPFAKE-ANALYSYS-SYSTEM.git
cd DEEPFAKE-ANALYSYS-SYSTEM
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# optional, better face detection (pins an old torch, so install without deps):
pip install --no-deps facenet-pytorch
# development / tests
pip install -r requirements-dev.txt && pytest
```

`ffmpeg` on PATH is recommended for audio-in-video. Without MTCNN the OpenCV
Haar detector is used; without either, the whole frame is analysed.

## Run

```bash
cp .env.example .env            # optional settings
streamlit run app.py            # UI: single, webcam, batch/CSV, compare, lab, signed report
python cli_scan.py -i suspect.mp4 -o reports --mode balanced
```

## Training and calibration workflow

Do this on a GPU box (Colab / Kaggle). Folder layout: `data/<name>/real/…` and
`data/<name>/fake/…`. Splits are group-aware (frames of one video never land in
both train and validation).

```bash
# 1. cache frozen-encoder embeddings (hub id or local dir; DINOv2/3, SigLIP2, CLIP)
python -m train.features --data data/ffpp --encoder <encoder-id> --out feats.npz
# 2. linear probe (seconds on CPU)
python -m train.train_probe --train feats.npz --out artifacts/probe.npz
#    optional: LayerNorm-only tuning (GenD-style) when the probe is not enough
python -m train.finetune_ln --data data/ffpp --encoder <encoder-id> --out artifacts
# 3. optional temporal head for video
python -m train.train_temporal --data data/ffpp_video --encoder <encoder-id> --out artifacts
# 4. set EMBED_MODEL_ID=<encoder-id or artifacts/finetuned_encoder> in .env, then calibrate
#    on a HELD-OUT set that was not used for training:
python -m train.fit_calibration --data data/calib --media image
python -m train.fit_calibration --data data/calib_video --media video
python -m train.fit_calibration --data data/calib_audio --media audio
# 5. evaluate on an unseen dataset / generator (cross-dataset is what matters)
python -m train.evaluate --data data/celebdf --media video --mode balanced --out eval.json
# 6. monitor drift (copy canary/manifest.example.json to canary/manifest.json)
python -m scripts.run_canary --set-baseline
python -m scripts.run_canary
```

Outputs land in `artifacts/` (`probe.npz`, `calibration.json`, `ood_*.npz`,
`fusion_weights.json`, temporal head). `fit_fusion_weights` in
`detector/fusion.py` can learn expert weights by logistic stacking.

For audio, fine-tune an XLS-R + AASIST style model externally and point
`AUDIO_MODEL_ID` at it.

## Configuration (env vars)

`IMAGE_MODEL_ID`, `EMBED_MODEL_ID`, `AUDIO_MODEL_ID`, `HF_TOKEN`,
`DEEPFAKE_ARTIFACT_DIR`, `DEEPFAKE_DEVICE`, `DEEPFAKE_ALPHA`,
`DEEPFAKE_TIER1_FRAMES`, `DEEPFAKE_FACE_SECONDS`. See `.env.example`.

## Layout

```
detector/   pipeline, models, calibration, fusion, triage, explain, stability,
            canary, report_generator, experts/{temporal,tamper,facesignals}
train/      data, features, train_probe, finetune_ln, train_temporal,
            fit_calibration, evaluate
scripts/    run_canary, verify_report
tests/      pytest suite (synthetic data, offline)
app.py  cli_scan.py
```

## Limitations/working on 

* Accuracy is unmeasured until you train, calibrate and evaluate on real data.
* Conformal guarantees assume test data resembles the calibration data;
  new generators break that, which is why the drift canary exists.
* Heuristic experts (rPPG, sync, geometry) are weak on compressed video.
* Occlusion heatmaps explain the model's behaviour, not ground truth about
  where the manipulation is; the faithfulness score is the sanity check.
* Detection results are probabilistic evidence, not proof.

Built with ❤️ OGmayur
