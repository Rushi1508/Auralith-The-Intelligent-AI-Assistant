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

# Streamlit only reloads changed modules under ui/; main.py (one level up) kept its
# old imports, so pushes never showed without a reboot. Drop our modules when code changes.
_ROOT = Path(__file__).resolve().parent.parent
_stamp = max(
    p.stat().st_mtime
    for p in [_ROOT / "main.py", *(_ROOT / "ui").rglob("*.py"), *(_ROOT / "auralith").rglob("*.py")]
)
if getattr(sys.modules.get("main"), "_code_stamp", None) != _stamp:
    for _name in [m for m in sys.modules if m.split(".")[0] in ("main", "ui", "auralith")]:
        del sys.modules[_name]

import main as _app  # noqa: E402

_app._code_stamp = _stamp
_app.main()
