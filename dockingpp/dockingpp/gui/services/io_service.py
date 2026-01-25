"""IO helpers for GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st


def ensure_dir(path: Path) -> Path:
    """Ensure that a directory exists."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def save_uploaded_file(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
    out_dir: Path,
    filename_hint: Optional[str] = None,
) -> Path:
    """Persist an uploaded file to disk."""

    ensure_dir(out_dir)
    filename = filename_hint or uploaded_file.name
    path = out_dir / filename
    with open(path, "wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return path

