from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pandas as pd
import streamlit as st

from music_detection.classical import predict_files as predict_classical_files
from music_detection.modern import predict_files as predict_modern_files


if not st.runtime.exists():
    if os.environ.get("SPECTRAGUARD_STREAMLIT_LAUNCHED") != "1":
        environment = os.environ.copy()
        environment["SPECTRAGUARD_STREAMLIT_LAUNCHED"] = "1"
        raise SystemExit(
            subprocess.call(
                [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())],
                env=environment,
            )
        )


st.set_page_config(
    page_title="SpectraGuard | AI Music Detection",
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
        .topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5.2rem; }
        .brand { display: flex; gap: .75rem; align-items: center; color: var(--ink); font-weight: 800; font-size: 1.05rem; }
        .brand-mark { display: grid; place-items: center; width: 31px; height: 31px; border: 1px solid var(--cyan); color: var(--cyan); border-radius: 9px; box-shadow: 0 0 24px rgba(98,228,230,.18); }
        .hero-copy { max-width: 700px; margin-bottom: 2.5rem; }
        .hero-copy p { color: var(--muted); font-size: 1.03rem; line-height: 1.7; max-width: 600px; }
        .accent { color: var(--cyan); }
        .hero-meta { display: flex; gap: 1.8rem; color: var(--muted); font-size: .78rem; margin-top: 1.5rem; }
        .hero-meta span { color: var(--ink); margin-left: .35rem; }
        .panel { background: linear-gradient(145deg, rgba(22,31,46,.86), rgba(13,19,29,.86)); border: 1px solid var(--line); border-radius: 20px; padding: 1.35rem; }
        .panel-title { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.2rem; }
        .panel-title h3 { margin: 0; font-size: 1rem; letter-spacing: -.02em; }
        .step { color: var(--muted); font: .68rem 'DM Mono', monospace; border: 1px solid var(--line); padding: .35rem .55rem; border-radius: 99px; }
        .drop-copy { text-align: center; padding: 3.2rem 1rem 2.8rem; border: 1px dashed rgba(98,228,230,.38); border-radius: 14px; background: rgba(98,228,230,.025); }
        .upload-icon { margin: auto auto 1rem; display: grid; place-items: center; width: 54px; height: 54px; border-radius: 16px; background: rgba(98,228,230,.1); color: var(--cyan); font-size: 1.5rem; }
        .drop-copy strong { display: block; font-size: 1rem; margin-bottom: .45rem; }
        .drop-copy span { color: var(--muted); font-size: .8rem; }
        .hint { color: var(--muted); font-size: .73rem; margin-top: .9rem; }
        .metric-card { background: var(--panel); border: 1px solid var(--line); padding: 1rem; border-radius: 14px; min-height: 105px; }
        .metric-label { color: var(--muted); font-size: .72rem; }
        .metric-value { font: 500 1.3rem 'DM Mono', monospace; margin-top: .7rem; }
        .disabled-card { opacity: .58; }
        .pipeline { display: flex; align-items: center; gap: .55rem; margin-top: 1.2rem; color: var(--muted); font: .68rem 'DM Mono', monospace; }
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
        [data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, disabled: bool = False) -> None:
    class_name = "metric-card disabled-card" if disabled else "metric-card"
    st.markdown(
        f'<div class="{class_name}"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


inject_styles()

st.markdown(
    """
    <div class="topbar">
      <div class="brand"><div class="brand-mark">◈</div> SPECTRAGUARD</div>
    </div>
    <div class="hero-copy">
      <div class="eyebrow">AI music detection · spectral analysis</div>
      <h1>Find the signal<br>inside the <span class="accent">sound.</span></h1>
      <p>Upload an audio track to inspect its spectral fingerprint. Our research pipeline is designed to surface the subtle artifacts that separate generated music from human performance.</p>
      <div class="hero-meta"><div>MODELS <span>SVM / RESNET-18</span></div><div>INPUT <span>WAV / MP3</span></div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.5, 1], gap="large")
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
        "Choose detection model",
        ["Classical SVM", "Modern ResNet-18"],
        horizontal=True,
        help="SVM uses MFCC and spectral features. ResNet-18 uses log-Mel spectrograms.",
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
            use_modern = selected_model == "Modern ResNet-18"
            model_path = Path("runs/m6_modern/resnet18.pt" if use_modern else "runs/m6_classical/svm.joblib")
            if not model_path.is_file():
                command = "Windows training command" if use_modern else "make m6-classical"
                st.error(f"Model artifact not found. Run the {command} first, then upload the files again.")
            else:
                with tempfile.TemporaryDirectory(prefix="spectraguard-") as temp_dir:
                    paths = []
                    for index, audio_file in enumerate(uploaded_file):
                        path = Path(temp_dir) / f"{index}{Path(audio_file.name).suffix.lower()}"
                        path.write_bytes(audio_file.getbuffer())
                        paths.append(path)
                    with st.spinner("Extracting spectral features and classifying files..."):
                        predict = predict_modern_files if use_modern else predict_classical_files
                        results = predict(model_path, paths)
                results.insert(1, "model", selected_model)
                results["ai_probability"] = (results["ai_probability"] * 100).round(2)
                st.session_state["analysis_results"] = results
    else:
        st.markdown('<div class="hint">Upload multiple files · Maximum file size: 200 MB each · For research use only</div>', unsafe_allow_html=True)
    if "analysis_results" in st.session_state:
        results = st.session_state["analysis_results"]
        st.markdown("#### Batch analysis results")
        st.dataframe(
            results,
            column_config={
                "ai_probability": st.column_config.ProgressColumn(
                    "AI probability (%)", min_value=0, max_value=100, format="%.2f%%"
                )
            },
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "Download results CSV",
            results.to_csv(index=False).encode("utf-8"),
            "spectraguard-results.csv",
            "text/csv",
            use_container_width=True,
        )
    st.markdown('<div class="pipeline"><b>UPLOAD</b><span class="arrow">→</span><span>PROCESSING</span><span class="arrow">→</span><span>RESULTS</span></div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel"><div class="panel-title"><h3>Analysis preview</h3><span class="step">02 / 03</span></div>', unsafe_allow_html=True)
    metric_card("AI PROBABILITY", "— %", disabled=True)
    st.write("")
    metric_card("CONFIDENCE", "Awaiting audio", disabled=True)
    st.write("")
    metric_card("FLAGGED SEGMENTS", "—", disabled=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer">
      <div>SPECTRAGUARD / SMAI RESEARCH PROJECT</div>
      <div>BUILD 0.1.0 &nbsp;·&nbsp; TEAM SARCOSUCHUS</div>
    </div>
    """,
    unsafe_allow_html=True,
)
