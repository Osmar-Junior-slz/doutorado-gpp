"""Reusable Streamlit UI components."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st


def download_json_button(
    label: str,
    path: Path,
    filename: Optional[str] = None,
    warn_missing: bool = True,
) -> None:
    """Render a download button for JSON files."""

    if not path.exists():
        if warn_missing:
            st.warning(f"{path.name} não encontrado.")
        return
    st.download_button(
        label,
        data=path.read_text(encoding="utf-8"),
        file_name=filename or path.name,
        mime="application/json",
    )


def download_binary_button(label: str, path: Path, mime: str, warn_missing: bool = True) -> None:
    """Render a download button for binary files."""

    if not path.exists():
        if warn_missing:
            st.warning(f"{path.name} não encontrado.")
        return
    st.download_button(
        label,
        data=path.read_bytes(),
        file_name=path.name,
        mime=mime,
    )
