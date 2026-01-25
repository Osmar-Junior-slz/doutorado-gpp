"""Dialog helpers for the GUI."""

from __future__ import annotations

import importlib.util


def choose_directory() -> str | None:
    """Open a native directory chooser dialog when available."""

    if importlib.util.find_spec("tkinter") is None:
        return None

    import tkinter as tk
    from tkinter import filedialog

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        directory = filedialog.askdirectory()
        root.destroy()
    except Exception:  # noqa: BLE001
        return None

    return directory or None

