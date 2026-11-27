"""Streamlit Community Cloud entrypoint (set it as the app's main file)."""

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Cloud keeps secrets in st.secrets; the UI reads env vars.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:  # no secrets file locally
    pass

from main import main  # noqa: E402

main()
