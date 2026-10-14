"""Streamlit rendering of an AnalysisResult (evidence, stability, heatmaps)."""
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.ui_components import render_result_card


def evidence_frame(res: dict) -> pd.DataFrame:
    rows = []
    for e in res.get("evidence", []):
        rows.append({
            "Evidence": e["name"] + (" (heuristic)" if e.get("heuristic") else ""),
            "Status": e["status"],
            "Fake log-odds": e["logit"],
            "P(fake) alone": e["p_fake_raw"],
            "Reliability": e["reliability"],
            "Note": e.get("reason") or "",
        })
    return pd.DataFrame(rows)


def render_result(res_dict: dict, artifacts: dict, key: str = "r"):
    """Full result view used by the single-file scanner."""
    render_result_card(res_dict)
    media = res_dict.get("media_type")
    q = res_dict.get("quality", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verdict", res_dict["verdict"])
    c2.metric("P(manipulated)", f"{res_dict['p_fake'] * 100:.1f}%")
    st_ = res_dict.get("stability") or {}
    c3.metric("Re-encoding stability",
              f"{st_['stability_score'] * 100:.0f}%" if st_.get("stability_score") is not None else "n/a")
    ex = res_dict.get("explanation") or {}
    c4.metric("Heatmap faithfulness",
              f"{ex['faithfulness']:+.2f}" if ex.get("faithfulness") is not None else "n/a")

    with st.expander("🧾 Plain-language summary", expanded=True):
        st.text(res_dict.get("narrative", ""))

    st.markdown("### 🧩 Evidence behind the verdict")
    st.dataframe(evidence_frame(res_dict), width="stretch", hide_index=True)
    contrib = (res_dict.get("fusion") or {}).get("contributions")
    if contrib:
        st.caption("Share of the fused decision per source: " +
                   ", ".join(f"{k} {v * 100:.0f}%" for k, v in contrib.items()))

    if media == "image":
        st.markdown("### 🖼️ Visual evidence")
        cols = st.columns(3)
        if artifacts.get("annotated") is not None:
            cols[0].image(artifacts["annotated"], caption="Detected faces", width="stretch")
        if artifacts.get("heatmap") is not None:
            cols[1].image(artifacts["heatmap"], width="stretch",
                          caption="Occlusion heatmap: red regions push the score towards 'fake'")
        if artifacts.get("tamper_map") is not None:
            cols[2].image(artifacts["tamper_map"], width="stretch",
                          caption="Noise-inconsistency map (heuristic)")
        with st.expander("Classic forensic views (ELA, FFT)"):
            c = st.columns(2)
            if artifacts.get("ela") is not None:
                c[0].image(artifacts["ela"], caption="Error Level Analysis", width="stretch")
            if artifacts.get("fft") is not None:
                c[1].image(artifacts["fft"], caption="2D FFT spectrum", width="stretch")
        if artifacts.get("face_crops"):
            cols = st.columns(min(len(artifacts["face_crops"]), 4))
            for i, im in enumerate(artifacts["face_crops"]):
                cols[i % 4].image(im, caption=f"Face region #{i + 1}", width="stretch")

    if media == "video":
        tl = res_dict.get("timeline") or []
        if tl:
            st.markdown("### 📈 Per-frame manipulation probability")
            df = pd.DataFrame(tl)
            fig = px.line(df, x="timestamp", y="fake_probability", markers=True,
                          labels={"timestamp": "Time (s)", "fake_probability": "P(fake)"})
            fig.add_hline(y=0.5, line_dash="dash", line_color="#94a3b8")
            fig.update_yaxes(range=[0, 1])
            fig.update_layout(paper_bgcolor="rgba(15,23,42,0.8)",
                              plot_bgcolor="rgba(15,23,42,0.8)", font_color="#e2e8f0")
            st.plotly_chart(fig, width="stretch", key=f"{key}_tl")
        segs = [s for s in res_dict.get("segments", []) if "end" in s and "logit" not in s and "fake_probability" not in s]
        if segs:
            st.info("Suspicious time ranges: " +
                    ", ".join(f"{s['start']:.1f}s-{s['end']:.1f}s" for s in segs))
        sus = artifacts.get("suspicious_frames") or []
        if sus:
            st.markdown("### 🖼️ Most suspicious sampled frames")
            cols = st.columns(min(len(sus), 4))
            for i, (t, im) in enumerate(sus):
                cols[i % 4].image(im, caption=f"t = {t:.1f}s", width="stretch")
        if artifacts.get("heatmap") is not None:
            st.image(artifacts["heatmap"], width=360,
                     caption="Occlusion heatmap of the most suspicious face")

    if media == "audio":
        tl = res_dict.get("timeline") or []
        if artifacts.get("mel") is not None:
            st.image(artifacts["mel"], width="stretch", caption="Mel spectrogram")
        if tl:
            df = pd.DataFrame(tl)
            fig = px.line(df, x="timestamp", y="fake_probability", markers=True,
                          labels={"timestamp": "Time (s)", "fake_probability": "P(synthetic)"})
            fig.add_hline(y=0.5, line_dash="dash", line_color="#94a3b8")
            fig.update_yaxes(range=[0, 1])
            fig.update_layout(paper_bgcolor="rgba(15,23,42,0.8)",
                              plot_bgcolor="rgba(15,23,42,0.8)", font_color="#e2e8f0")
            st.plotly_chart(fig, width="stretch", key=f"{key}_tl")

    with st.expander("📏 Media quality and provenance"):
        st.json({"quality": q, "provenance": res_dict.get("provenance"),
                 "sha256": res_dict.get("sha256")})
    meta = res_dict.get("metadata")
    if meta:
        with st.expander("📄 File metadata (EXIF / headers)"):
            for w in meta.get("warnings", []):
                st.error(w)
            for n in meta.get("notes", []):
                st.caption(n)
            st.dataframe(pd.DataFrame([{"Property": k, "Value": str(v)} for k, v in meta.items()
                                       if k not in ("exif_data", "warnings", "notes")]),
                         width="stretch", hide_index=True)
    with st.expander("⏱️ Timings"):
        st.json(res_dict.get("timings", {}))
