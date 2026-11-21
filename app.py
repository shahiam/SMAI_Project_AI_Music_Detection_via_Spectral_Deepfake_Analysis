from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from music_detection.classical import predict_files as predict_classical_files
from music_detection.modern import predict_files as predict_modern_files


if not st.runtime.exists():
    if os.environ.get("SARCOSPECTUS_STREAMLIT_LAUNCHED") != "1":
        environment = os.environ.copy()
        environment["SARCOSPECTUS_STREAMLIT_LAUNCHED"] = "1"
        raise SystemExit(
            subprocess.call(
                [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())],
                env=environment,
            )
        )


# Which trained models the dashboard serves. Switch with the SARCOSPECTUS_PROFILE env var.
#   legacy      - original 44.1 kHz M6 models with the hand-set cutoffs (current default)
#   bandlimited - 16 kHz shortcut-free models; thresholds/ensemble weight tuned on validation
#                 (better on M6, but misses modern Suno-style covers; kept for later work)
PROFILES = {
    "legacy": {
        "modern": ["runs/m6_modern/resnet18.pt", "runs/modern/resnet18.pt"],
        "classical": ["runs/m6_classical/svm.joblib", "runs/classical/svm.joblib"],
        "modern_threshold": 0.70,
        "classical_threshold": 0.65,
        "ensemble": {"modern_weight": 0.40, "threshold": 0.55},
    },
    "bandlimited": {
        "modern": ["runs/m6_bl_modern/resnet18.pt"],
        "classical": ["runs/m6_bl_classical/svm.joblib"],
        "modern_threshold": None,  # None = use the threshold stored with the model
        "classical_threshold": None,
        "ensemble": "runs/m6_bl_ensemble/ensemble.json",
    },
}
ACTIVE_PROFILE = os.environ.get("SARCOSPECTUS_PROFILE", "legacy")
PROFILE = PROFILES[ACTIVE_PROFILE]


def first_existing(kind: str) -> Path:
    for candidate in PROFILE[kind]:
        if Path(candidate).is_file():
            return Path(candidate)
    return Path(PROFILE[kind][-1])


def load_ensemble_config() -> dict | None:
    """Blend weight + threshold: fixed for legacy, fitted on validation for bandlimited."""
    config = PROFILE["ensemble"]
    if isinstance(config, dict):
        return config
    if Path(config).is_file():
        return json.loads(Path(config).read_text(encoding="utf-8"))
    return None


st.set_page_config(
    page_title="Sarcospectus | AI Music Deepfake Detection",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {
            --ink: #f4f7fb;
            --muted: #8d9aad;
            --line: rgba(165, 185, 210, .15);
            --panel: rgba(19, 27, 40, .78);
            --panel-strong: #151f2f;
            --cyan: #62e4e6;
            --violet: #9a7cff;
            --green: #70dfb2;
            --coral: #ff6b6b;
        }

        .stApp {
            background:
                radial-gradient(circle at 82% 0%, rgba(80, 105, 220, .16), transparent 33rem),
                radial-gradient(circle at 10% 12%, rgba(44, 192, 193, .08), transparent 25rem),
                #090d14;
            color: var(--ink);
            font-family: 'Manrope', sans-serif;
        }
        .block-container { max-width: 1240px; padding: 2rem 3rem 4rem; }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { display: none; }
        h1, h2, h3, p { font-family: 'Manrope', sans-serif; }
        h1 { letter-spacing: -.055em; font-weight: 800; font-size: clamp(2.4rem, 5vw, 4.65rem); line-height: .98; }
        h2 { letter-spacing: -.035em; font-size: 1.65rem; }
        .eyebrow, .mono { font-family: 'DM Mono', monospace; text-transform: uppercase; letter-spacing: .12em; }
        .eyebrow { color: var(--cyan); font-size: .72rem; }
        .topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 3.5rem; }
        .brand { display: flex; gap: .75rem; align-items: center; color: var(--ink); font-weight: 800; font-size: 1.05rem; }
        .brand-mark { display: grid; place-items: center; width: 31px; height: 31px; border: 1px solid var(--cyan); color: var(--cyan); border-radius: 9px; box-shadow: 0 0 24px rgba(98,228,230,.18); }
        .hero-copy { max-width: 700px; margin-bottom: 2.2rem; }
        .hero-copy p { color: var(--muted); font-size: 1.03rem; line-height: 1.7; max-width: 600px; }
        .accent { color: var(--cyan); }
        .hero-meta { display: flex; gap: 1.8rem; color: var(--muted); font-size: .78rem; margin-top: 1.5rem; }
        .hero-meta span { color: var(--ink); margin-left: .35rem; }
        .panel { background: linear-gradient(145deg, rgba(22,31,46,.86), rgba(13,19,29,.86)); border: 1px solid var(--line); border-radius: 20px; padding: 1.35rem; margin-bottom: 1.2rem; }
        .panel-title { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.2rem; }
        .panel-title h3 { margin: 0; font-size: 1rem; letter-spacing: -.02em; }
        .step { color: var(--muted); font: .68rem 'DM Mono', monospace; border: 1px solid var(--line); padding: .35rem .55rem; border-radius: 99px; }
        .drop-copy { text-align: center; padding: 2.5rem 1rem 2.2rem; border: 1px dashed rgba(98,228,230,.38); border-radius: 14px; background: rgba(98,228,230,.025); }
        .upload-icon { margin: auto auto .8rem; display: grid; place-items: center; width: 50px; height: 50px; border-radius: 16px; background: rgba(98,228,230,.1); color: var(--cyan); font-size: 1.4rem; }
        .drop-copy strong { display: block; font-size: 1rem; margin-bottom: .45rem; }
        .drop-copy span { color: var(--muted); font-size: .8rem; }
        .hint { color: var(--muted); font-size: .73rem; margin-top: .9rem; }
        .metric-card { background: var(--panel); border: 1px solid var(--line); padding: 1.1rem; border-radius: 14px; min-height: 105px; }
        .metric-label { color: var(--muted); font-size: .72rem; }
        .metric-value { font: 600 1.35rem 'DM Mono', monospace; margin-top: .6rem; }
        .metric-sub { font-size: .75rem; color: var(--muted); margin-top: .3rem; }
        .disabled-card { opacity: .58; }
        .badge-ai { color: var(--coral); background: rgba(255,107,107,.15); padding: .2rem .55rem; border-radius: 6px; font-size: .78rem; font-weight: 700; border: 1px solid rgba(255,107,107,.3); }
        .badge-human { color: var(--green); background: rgba(112,223,178,.15); padding: .2rem .55rem; border-radius: 6px; font-size: .78rem; font-weight: 700; border: 1px solid rgba(112,223,178,.3); }
        .pipeline { display: flex; align-items: center; gap: .55rem; margin-top: 1.5rem; color: var(--muted); font: .68rem 'DM Mono', monospace; }
        .pipeline b { color: var(--cyan); font-weight: 400; }
        .arrow { color: #475569; }
        .footer { border-top: 1px solid var(--line); margin-top: 5rem; padding-top: 1.2rem; color: #647286; font-size: .72rem; display: flex; justify-content: space-between; }
        .stButton > button { border-radius: 10px; border: 1px solid rgba(98,228,230,.38); background: rgba(98,228,230,.1); color: var(--cyan); font-weight: 700; }
        .stButton > button:hover { border-color: var(--cyan); color: white; }
        [data-testid="stFileUploader"] { margin: .45rem 0 0 -1rem; }
        [data-testid="stFileUploader"] section { border: 0; background: transparent; padding: 0; min-height: 0; }
        [data-testid="stFileUploader"] section > div { min-height: 0; padding: 0; }
        [data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
        [data-testid="stFileUploader"] button { border-radius: 9px; background: var(--cyan); color: #081017; border: 0; font-weight: 800; padding: .55rem 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sublabel: str = "", disabled: bool = False, badge_html: str = "") -> None:
    class_name = "metric-card disabled-card" if disabled else "metric-card"
    badge_part = f" {badge_html}" if badge_html else ""
    sub_part = f'<div class="metric-sub">{sublabel}</div>' if sublabel else ""
    st.markdown(
        f'<div class="{class_name}"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}{badge_part}</div>{sub_part}</div>',
        unsafe_allow_html=True,
    )


inject_styles()

st.markdown(
    """
    <div class="topbar">
      <div class="brand"><div class="brand-mark">◈</div> SARCOSPECTUS</div>
    </div>
    <div class="hero-copy">
      <div class="eyebrow">AI music detection · spectral deepfake analysis · team sarcosuchus</div>
      <h1>Find the signal<br>lurking beneath the <span class="accent">sound.</span></h1>
      <p>Upload audio tracks to inspect their sub-surface spectral and acoustic deepfake fingerprint. Powered by full-track window aggregation, calibrated low-FPR decision rules, and hybrid neural-acoustic ensembling.</p>
      <div class="hero-meta"><div>MODELS <span>RESNET-18 / SVM / HYBRID</span></div><div>SCANNING <span>FULL-TRACK WINDOWING</span></div><div>CALIBRATION <span>LOW FALSE-POSITIVE TUNED</span></div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.45, 1.05], gap="large")

with left:
    upload_header, upload_button, upload_step = st.columns([1.45, .95, .75], vertical_alignment="center")
    with upload_header:
        st.markdown('<h3 style="margin: 0;">Upload audio sample</h3>', unsafe_allow_html=True)
    with upload_button:
        st.markdown('<div style="height: .45rem;"></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload audio files",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
    with upload_step:
        st.markdown('<div class="step" style="text-align: center;">01 / INPUT</div>', unsafe_allow_html=True)

    selected_model = st.radio(
        "Choose detection architecture",
        ["Hybrid Ensemble (Modern + Classical)", "Modern ResNet-18", "Classical SVM"],
        horizontal=True,
        help="Hybrid Ensemble averages the ResNet-18 and SVM probabilities. The weight and cutoff are fitted on the M6 validation split.",
    )

    st.markdown(
        '<div style="font-size: .78rem; color: var(--muted); margin-top: .4rem; margin-bottom: 1.1rem;">'
        '⚡ <b style="color: var(--cyan);">Optimized Parameters Auto-Applied</b>: 5.0s windowing · Full-track aggregation · Calibrated threshold (protects human tracks from false positives)'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel">
          <div class="drop-copy">
            <div class="upload-icon">↑</div>
            <strong>Choose audio files to begin</strong>
            <span>Upload one or more WAV, MP3, FLAC, OGG, or M4A files</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if uploaded_file:
        st.success(f"{len(uploaded_file)} file(s) ready for analysis")
        with st.expander("Preview uploaded audio", expanded=False):
            for audio_file in uploaded_file:
                st.write(audio_file.name)
                st.audio(audio_file)

        if st.button("Analyze selected files  →", use_container_width=True, type="primary"):
            modern_path = first_existing("modern")
            classical_path = first_existing("classical")
            ensemble_config = load_ensemble_config()

            with tempfile.TemporaryDirectory(prefix="sarcospectus-") as temp_dir:
                paths = []
                file_name_map = {}
                for index, audio_file in enumerate(uploaded_file):
                    ext = Path(audio_file.name).suffix.lower() or ".wav"
                    path = Path(temp_dir) / f"{index}{ext}"
                    path.write_bytes(audio_file.getbuffer())
                    paths.append(path)
                    file_name_map[path.name] = audio_file.name

                with st.spinner(f"Analyzing full track(s) using {selected_model}..."):
                    if selected_model == "Hybrid Ensemble (Modern + Classical)":
                        # Weight and cutoff come from the active profile: fixed 0.40/0.55 for legacy,
                        # fitted on validation (ensemble.json) for bandlimited; 0.5/0.5 if missing.
                        modern_weight = float(ensemble_config["modern_weight"]) if ensemble_config else 0.5
                        ens_thresh = float(ensemble_config["threshold"]) if ensemble_config else 0.5
                        m_df = predict_modern_files(
                            modern_path,
                            paths,
                            window_duration=5.0,
                            hop_duration=5.0,
                            aggregation="mean",
                            threshold=PROFILE["modern_threshold"],
                            detailed=True,
                        )
                        c_df = predict_classical_files(
                            classical_path,
                            paths,
                            window_duration=5.0,
                            hop_duration=5.0,
                            aggregation="mean",
                            threshold=PROFILE["classical_threshold"],
                            detailed=True,
                        )

                        ens_rows = []
                        for i in range(len(m_df)):
                            m_row = m_df.iloc[i]
                            c_row = c_df.iloc[i]
                            m_prob = float(m_row["ai_probability"])
                            c_prob = float(c_row["ai_probability"])
                            ens_prob = modern_weight * m_prob + (1.0 - modern_weight) * c_prob

                            is_ai = bool(ens_prob >= ens_thresh)
                            pred = "AI-generated" if is_ai else "Human"

                            if ens_prob >= ens_thresh:
                                conf = (ens_prob - ens_thresh) / max(1.0 - ens_thresh, 1e-6)
                            else:
                                conf = (ens_thresh - ens_prob) / max(ens_thresh, 1e-6)
                            conf = float(np.clip(conf, 0.0, 1.0))

                            # Blend segments
                            m_segs = m_row["segments"]
                            c_segs = c_row["segments"]
                            blended_segs = []
                            for idx in range(min(len(m_segs), len(c_segs))):
                                s_prob = (
                                    modern_weight * m_segs[idx]["probability"]
                                    + (1.0 - modern_weight) * c_segs[idx]["probability"]
                                )
                                blended_segs.append(
                                    {
                                        "window_index": idx,
                                        "start_sec": m_segs[idx]["start_sec"],
                                        "end_sec": m_segs[idx]["end_sec"],
                                        "probability": s_prob,
                                        "flagged": bool(s_prob >= ens_thresh),
                                    }
                                )

                            flagged_count = sum(1 for s in blended_segs if s["flagged"])
                            ens_rows.append(
                                {
                                    "filename": file_name_map.get(m_row["filename"], m_row["filename"]),
                                    "model": "Hybrid Ensemble",
                                    "ai_probability": ens_prob,
                                    "prediction": pred,
                                    "confidence": conf,
                                    "flagged_windows": flagged_count,
                                    "total_windows": len(blended_segs),
                                    "threshold": ens_thresh,
                                    "segments": blended_segs,
                                }
                            )
                        detailed_df = pd.DataFrame(ens_rows)
                        threshold_used = ens_thresh

                    elif selected_model == "Modern ResNet-18":
                        detailed_df = predict_modern_files(
                            modern_path,
                            paths,
                            window_duration=5.0,
                            hop_duration=5.0,
                            aggregation="mean",
                            threshold=PROFILE["modern_threshold"],  # None = tuned value stored in the checkpoint
                            detailed=True,
                        )
                        threshold_used = float(detailed_df["threshold"].iloc[0])
                        detailed_df["filename"] = detailed_df["filename"].map(lambda f: file_name_map.get(f, f))
                        detailed_df.insert(1, "model", "Modern ResNet-18")

                    else:  # Classical SVM
                        detailed_df = predict_classical_files(
                            classical_path,
                            paths,
                            window_duration=5.0,
                            hop_duration=5.0,
                            aggregation="mean",
                            threshold=PROFILE["classical_threshold"],  # None = tuned value stored with the model
                            detailed=True,
                        )
                        threshold_used = float(detailed_df["threshold"].iloc[0])
                        detailed_df["filename"] = detailed_df["filename"].map(lambda f: file_name_map.get(f, f))
                        detailed_df.insert(1, "model", "Classical SVM")

            st.session_state["analysis_detailed"] = detailed_df
            st.session_state["threshold_used"] = threshold_used
            st.session_state["model_used"] = selected_model
            st.session_state["model_paths"] = f"{modern_path} · {classical_path}"
    else:
        st.markdown(
            '<div class="hint">Upload audio files · Full-track temporal windowing scans beyond the initial 5 seconds · For research use only</div>',
            unsafe_allow_html=True,
        )

    if "analysis_detailed" in st.session_state:
        df = st.session_state["analysis_detailed"]
        st.markdown("#### Batch analysis results")

        summary_df = df[["filename", "model", "ai_probability", "prediction", "confidence", "flagged_windows", "total_windows"]].copy()
        summary_df["ai_probability"] = (summary_df["ai_probability"] * 100).round(2)
        summary_df["confidence"] = (summary_df["confidence"] * 100).round(1).astype(str) + "%"
        summary_df["flagged_ratio"] = summary_df["flagged_windows"].astype(str) + " / " + summary_df["total_windows"].astype(str)
        summary_display = summary_df[["filename", "model", "ai_probability", "prediction", "confidence", "flagged_ratio"]]

        st.dataframe(
            summary_display,
            column_config={
                "ai_probability": st.column_config.ProgressColumn(
                    "AI probability (%)", min_value=0, max_value=100, format="%.2f%%"
                ),
                "flagged_ratio": st.column_config.TextColumn("Flagged Windows"),
            },
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download results CSV",
            summary_df.to_csv(index=False).encode("utf-8"),
            "sarcospectus-results.csv",
            "text/csv",
            use_container_width=True,
        )

        # Full-track Temporal Window Breakdown
        st.markdown("#### ⏱️ Full-Track Temporal Window Breakdown")
        track_names = list(df["filename"])
        selected_name = st.selectbox("Inspect track timeline", track_names, index=0)
        selected_row = df[df["filename"] == selected_name].iloc[0]

        segments = selected_row["segments"]
        timeline_data = pd.DataFrame(
            [
                {
                    "Time (seconds)": f"{s['start_sec']:.1f}s - {s['end_sec']:.1f}s",
                    "Timestamp": s["start_sec"],
                    "AI Probability (%)": round(s["probability"] * 100, 2),
                    "Status": "⚠️ Flagged" if s["flagged"] else "✓ Clean",
                }
                for s in segments
            ]
        )

        chart_df = timeline_data.set_index("Time (seconds)")["AI Probability (%)"]
        st.area_chart(chart_df, color="#62e4e6")
        st.caption(
            f"Decision threshold cutoff: **{st.session_state['threshold_used']*100:.1f}%** "
            f"· profile `{ACTIVE_PROFILE}` · models: `{st.session_state.get('model_paths', '')}`"
        )

        with st.expander("View window-by-window details", expanded=False):
            st.dataframe(
                timeline_data[["Time (seconds)", "AI Probability (%)", "Status"]],
                hide_index=True,
                use_container_width=True,
            )

    st.markdown(
        '<div class="pipeline"><b>UPLOAD</b><span class="arrow">→</span><span>FULL-TRACK WINDOWING</span><span class="arrow">→</span><span>CALIBRATED RESULTS</span></div>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="panel"><div class="panel-title"><h3>Analysis preview</h3><span class="step">02 / 03</span></div>', unsafe_allow_html=True)

    if "analysis_detailed" in st.session_state:
        df = st.session_state["analysis_detailed"]
        current_name = selected_name if "selected_name" in locals() else df.iloc[0]["filename"]
        track_row = df[df["filename"] == current_name].iloc[0]

        prob_pct = track_row["ai_probability"] * 100
        is_ai = track_row["prediction"] == "AI-generated"
        badge = '<span class="badge-ai">AI-GENERATED</span>' if is_ai else '<span class="badge-human">HUMAN</span>'

        metric_card(
            "AGGREGATED AI PROBABILITY",
            f"{prob_pct:.1f}%",
            sublabel=f"Cutoff: {st.session_state['threshold_used']*100:.0f}% · {st.session_state.get('model_used', 'Ensemble')}",
            badge_html=badge,
        )
        st.write("")

        conf_pct = track_row["confidence"] * 100
        conf_tier = "High" if conf_pct >= 70 else ("Moderate" if conf_pct >= 40 else "Borderline")
        metric_card(
            "CONFIDENCE",
            f"{conf_pct:.1f}%",
            sublabel=f"{conf_tier} separation from decision boundary",
        )
        st.write("")

        flagged = track_row["flagged_windows"]
        total = track_row["total_windows"]
        flag_pct = (flagged / max(1, total)) * 100
        metric_card(
            "FLAGGED SEGMENTS",
            f"{flagged} / {total} windows",
            sublabel=f"{flag_pct:.1f}% of track duration flagged",
        )
    else:
        metric_card("AI PROBABILITY", "— %", disabled=True)
        st.write("")
        metric_card("CONFIDENCE", "Awaiting audio", disabled=True)
        st.write("")
        metric_card("FLAGGED SEGMENTS", "—", disabled=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer">
      <div>SARCOSPECTUS / SMAI RESEARCH PROJECT</div>
      <div>BUILD 0.2.0 &nbsp;·&nbsp; TEAM SARCOSUCHUS &nbsp;·&nbsp; FULL-TRACK SPECTRAL ANALYSIS</div>
    </div>
    """,
    unsafe_allow_html=True,
)
