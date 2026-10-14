import json
import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from detector.canary import load_last as load_canary
from detector.pipeline import get_pipeline, infer_media_type
from detector.report_generator import (
    build_payload, generate_html_report, sign_payload,
)
from utils.result_view import render_result
from utils.ui_components import (
    apply_custom_theme, render_header, render_result_card,
)
from utils.visualizers import generate_ela_image, generate_fft_spectrum

st.set_page_config(
    page_title="🛡️ Deepfake Forensic Intelligence Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_custom_theme()
render_header()

pipe = get_pipeline()

# === SIDEBAR ===============================================================
st.sidebar.markdown("### ⚙️ Decision settings")
mode_label = st.sidebar.radio(
    "Operating mode",
    ["Balanced", "Strict (fewest false alarms)", "Sensitive (catch more fakes)"],
    help="Controls how much evidence is needed before calling something Fake. "
         "Once calibrated, Strict targets about 1% false alarms on real media, "
         "Balanced about 5%, Sensitive about 10%.",
)
mode = {"Balanced": "balanced", "Strict (fewest false alarms)": "strict",
        "Sensitive (catch more fakes)": "sensitive"}[mode_label]
explain = st.sidebar.checkbox("Explanations (heatmaps)", value=True,
                              help="Occlusion heatmap + faithfulness check. Slower.")
stability = st.sidebar.checkbox("Re-encoding stability test", value=True,
                                help="Replays the media through JPEG/blur/noise and "
                                     "checks the verdict does not flip.")
audio_chunk = st.sidebar.slider("Audio chunk (s)", 1, 5, 3)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 System status")
status = pipe.status()
if not status["any_calibrated"]:
    st.sidebar.warning(
        "Not calibrated yet. Verdicts use a conservative fallback and the "
        "probabilities are rough guides. See README → *Training and "
        "calibration*.")
else:
    st.sidebar.success("Calibrated: " + ", ".join(
        f"{k} (n={v['n_cal']})" for k, v in status["calibrators"].items() if v["fitted"]))
canary = load_canary()
if canary and canary.get("freshness"):
    fr = canary["freshness"]
    badge = {"fresh": "🟢", "watch": "🟡", "stale": "🔴"}.get(fr["status"], "⚪")
    st.sidebar.caption(f"{badge} Model freshness: **{fr['status']}** - {fr['reason']}")
else:
    st.sidebar.caption("⚪ Model freshness: no canary run yet (`python -m scripts.run_canary`).")
with st.sidebar.expander("Backends"):
    st.write({"device": status["device"], "image": status["image_backends"],
              "audio": status["audio_backend"],
              "learned fusion weights": status["learned_fusion_weights"],
              "learned temporal head": status["learned_temporal_head"]})


# === helpers ================================================================
def analyze(source, media, name, *, do_explain=None, do_stab=None):
    """Run the pipeline and store the result for the report tab."""
    try:
        res = pipe.analyze(
            source, media_type=media, filename=name, mode=mode,
            explain=explain if do_explain is None else do_explain,
            stability=stability if do_stab is None else do_stab,
            audio_chunk_s=audio_chunk,
        )
    except Exception as e:  # surface model-download / decode problems clearly
        msg = str(e)
        st.error(f"Analysis failed: {msg}")
        if any(k in msg.lower() for k in ("huggingface", "connection", "offline", "401", "403")):
            st.info("Model download failed. Check your network and, for gated "
                    "models, set HF_TOKEN in a .env file.")
        return None
    return res


def remember(res):
    st.session_state["latest"] = {"result": res.to_dict(), "artifacts": res.artifacts}


# === TABS ===================================================================
tab_single, tab_cam, tab_batch, tab_compare, tab_lab, tab_report = st.tabs([
    "🎯 Single File Scanner", "📷 Live Webcam Scanner", "⚡ Batch Forensic Suite",
    "🆚 Dual File Comparison", "🔬 Deep Forensic Lab", "📄 Signed Report Export",
])

with tab_single:
    st.markdown('<div class="section-title">📁 Select media for forensic inspection</div>',
                unsafe_allow_html=True)
    option = st.radio("Media category", ["Image", "Video (multimodal)", "Audio"], horizontal=True)
    spec = {
        "Image": ("image", ["jpg", "jpeg", "png", "webp"]),
        "Video (multimodal)": ("video", ["mp4", "avi", "mov", "mkv", "webm"]),
        "Audio": ("audio", ["wav", "mp3", "flac", "ogg", "m4a"]),
    }[option]
    up = st.file_uploader(f"Upload {spec[0]}", type=spec[1], key=f"single_{spec[0]}")
    if up is not None:
        data = up.getvalue()
        st.markdown("---")
        if spec[0] == "video":
            st.video(data)
        elif spec[0] == "audio":
            st.audio(data)
        elif spec[0] == "image":
            st.image(data, width=420)
        with st.spinner("🔬 Running the evidence cascade..."):
            res = analyze(data, spec[0], up.name)
        if res is not None:
            remember(res)
            render_result(res.to_dict(), res.artifacts, key=f"single_{spec[0]}")

with tab_cam:
    st.markdown('<div class="section-title">📷 Camera snapshot inspector</div>',
                unsafe_allow_html=True)
    st.caption("A single snapshot cannot show temporal evidence (pulse, lip-sync); "
               "use the video scanner for those.")
    pic = st.camera_input("Take a snapshot for a forensic scan")
    if pic is not None:
        with st.spinner("🔍 Analyzing camera frame..."):
            res = analyze(pic.getvalue(), "image", "camera_snapshot.png")
        if res is not None:
            remember(res)
            render_result(res.to_dict(), res.artifacts, key="cam")

with tab_batch:
    st.markdown('<div class="section-title">⚡ Batch scanning</div>', unsafe_allow_html=True)
    files = st.file_uploader(
        "Upload multiple files", accept_multiple_files=True, key="batch",
        type=["jpg", "jpeg", "png", "webp", "wav", "mp3", "flac", "mp4", "avi", "mov"])
    if files and st.button("🚀 Start batch analysis", type="primary"):
        rows, bar = [], st.progress(0.0)
        for i, f in enumerate(files):
            media = infer_media_type(f.name)
            res = analyze(f.getvalue(), media, f.name, do_explain=False, do_stab=False) if media else None
            if res is not None:
                top = max((e for e in res.evidence if e["status"] == "used"),
                          key=lambda e: abs(e["logit"] or 0), default=None)
                rows.append({
                    "File": f.name, "Type": media, "Verdict": res.verdict,
                    "P(fake)": round(res.p_fake, 3),
                    "Confidence": None if res.confidence is None else round(res.confidence, 3),
                    "Path": res.tier, "Calibrated": res.calibrated,
                    "Strongest evidence": top["name"] if top else "",
                    "Warnings": "; ".join(res.warnings)[:200],
                })
            else:
                rows.append({"File": f.name, "Type": media or "unsupported", "Verdict": "error"})
            bar.progress((i + 1) / len(files))
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("📥 Download CSV", df.to_csv(index=False), "batch_results.csv", "text/csv")

with tab_compare:
    st.markdown('<div class="section-title">🆚 Side-by-side inspector</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    fa = c1.file_uploader("Image A (reference)", type=["jpg", "jpeg", "png", "webp"], key="file_a")
    fb = c2.file_uploader("Image B (suspect)", type=["jpg", "jpeg", "png", "webp"], key="file_b")
    if fa is not None and fb is not None:
        with st.spinner("🔬 Comparing..."):
            ra = analyze(fa.getvalue(), "image", fa.name)
            rb = analyze(fb.getvalue(), "image", fb.name)
        for col, r, name in ((c1, ra, fa.name), (c2, rb, fb.name)):
            with col:
                if r is None:
                    continue
                d = r.to_dict()
                render_result_card(d)
                st.image(r.artifacts["annotated"], caption=name, width="stretch")
                if r.artifacts.get("heatmap") is not None:
                    st.image(r.artifacts["heatmap"], caption="Occlusion heatmap", width="stretch")
                st.caption(d["narrative"])

with tab_lab:
    st.markdown('<div class="section-title">🔬 Classic forensic views (educational)</div>',
                unsafe_allow_html=True)
    st.info("ELA and FFT are *visual aids*: they do not feed the verdict by themselves and "
            "can mislead on re-compressed images. The decision uses the evidence table.")
    lab = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="lab_img")
    if lab is not None:
        im = Image.open(lab).convert("RGB")
        q = st.slider("JPEG quality (ELA)", 50, 98, 90)
        s = st.slider("ELA brightness scale", 5, 30, 15)
        a, b, c = st.columns(3)
        a.image(im, caption="Original", width="stretch")
        b.image(generate_ela_image(im, quality=q, scale=s), caption=f"ELA (Q={q})", width="stretch")
        c.image(generate_fft_spectrum(im), caption="2D FFT magnitude", width="stretch")

with tab_report:
    st.markdown('<div class="section-title">📄 Signed forensic report</div>', unsafe_allow_html=True)
    latest = st.session_state.get("latest")
    if not latest:
        st.warning("No analysis yet. Scan a file in the Single File Scanner first.")
    else:
        d, arts = latest["result"], latest["artifacts"]
        st.markdown(f"**File:** `{d['filename']}` ({d['media_type']}) · **Verdict:** `{d['verdict']}` "
                    f"· **P(fake):** `{d['p_fake'] * 100:.1f}%`")
        payload = build_payload(d)
        signed = {"payload": payload, "signature": sign_payload(payload)}
        html = generate_html_report(d, arts, signed)
        c1, c2 = st.columns(2)
        c1.download_button("📥 HTML report", html, f"forensic_report_{d['filename']}.html",
                           "text/html", type="primary")
        c2.download_button("📥 Signed JSON report", json.dumps(signed, indent=2),
                           f"forensic_report_{d['filename']}.json", "application/json")
        st.caption("The JSON is signed (Ed25519). Check it later with "
                   "`python -m scripts.verify_report <file>.json`.")
        components.html(html, height=520, scrolling=True)

st.markdown("---")
st.caption("Built by OGmayur. Reports support human review; they are not proof. "
           "Detectors can be wrong, especially on generators they were not calibrated on.")
