"""Launcher for the Streamlit dashboard.

The dashboard code lives in src/music_detection/shared/output/dashboard.py. This file keeps the
usual commands working from the repo root:
    streamlit run app.py      (Windows: .\\.venv-win\\Scripts\\streamlit.exe run app.py)
    python app.py
    make dashboard
"""
import runpy

runpy.run_module("music_detection.shared.output.dashboard", run_name="__main__")
