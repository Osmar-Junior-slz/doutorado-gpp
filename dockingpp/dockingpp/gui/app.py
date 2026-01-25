"""Streamlit GUI for dockingpp."""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from dockingpp.data.io import load_config
from dockingpp.pipeline.run import Config, run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def init_state() -> None:
    """Initialize Streamlit session state."""

    if "screen" not in st.session_state:
        st.session_state.screen = "home"
    if "default_out_dir" not in st.session_state:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.default_out_dir = f"runs/run_{timestamp}"


def save_upload(upload: st.runtime.uploaded_file_manager.UploadedFile, dest_dir: Path) -> str:
    """Persist an uploaded file to disk and return its path."""

    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / upload.name
    with open(path, "wb") as handle:
        handle.write(upload.getbuffer())
    return str(path)


def parse_metrics(path: Path) -> Dict[str, Any]:
    """Parse metrics JSONL into a summary dictionary."""

    summary: Dict[str, Any] = {}
    if not path.exists():
        return summary

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            name = record.get("name")
            if not name:
                continue
            summary[name] = record.get("value")
    return summary


def parse_metrics_series(path: Path) -> list[dict[str, Any]]:
    """Parse metrics JSONL into a list of points for charting."""

    series: list[dict[str, Any]] = []
    if not path.exists():
        return series

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            name = record.get("name")
            if name not in {"best_score_cheap", "best_score"}:
                continue
            step = record.get("step")
            if step is None:
                step = record.get("generation")
            if step is None:
                continue
            series.append({"step": step, "score": record.get("value")})
    return series


def render_home() -> None:
    """Render the landing screen."""

    st.title("Docking Reduce")
    st.subheader("Protein–peptide blind docking — experimental GUI")
    if st.button("New Docking Experiment"):
        st.session_state.screen = "experiment"
        st.rerun()


def build_config_path(
    selection: str,
    uploaded_config: Optional[st.runtime.uploaded_file_manager.UploadedFile],
    temp_dir: Path,
) -> Path:
    """Resolve configuration path based on UI selection."""

    if selection == "Upload YAML" and uploaded_config is not None:
        return Path(save_upload(uploaded_config, temp_dir))
    return DEFAULT_CONFIG_PATH


def render_experiment() -> None:
    """Render the experiment screen."""

    st.header("Docking Experiment")
    if st.button("Back to Home"):
        st.session_state.screen = "home"
        st.rerun()

    receptor_upload = st.file_uploader("Receptor (.pdb)", type=["pdb"])
    peptide_upload = st.file_uploader("Peptide (.pdb)", type=["pdb"])

    out_dir = st.text_input("Output directory", value=st.session_state.default_out_dir, key="out_dir")

    st.subheader("Experiment Mode")
    mode = st.radio(
        "Select mode",
        options=["Single Run", "Compare (Full vs Reduced)"],
        index=0,
    )
    top_pockets_value = None
    if mode == "Compare (Full vs Reduced)":
        top_pockets_value = st.slider("Top pockets (reduced run)", min_value=1, max_value=20, value=5)

    config_choice = st.radio(
        "Configuration",
        options=["Default (configs/default.yaml)", "Upload YAML"],
        index=0,
    )
    uploaded_config = None
    if config_choice == "Upload YAML":
        uploaded_config = st.file_uploader("Config YAML", type=["yaml", "yml"])

    if st.button("Run"):
        if receptor_upload is None or peptide_upload is None:
            st.error("Please upload receptor and peptide PDB files before running.")
            return
        if config_choice == "Upload YAML" and uploaded_config is None:
            st.error("Please upload a YAML configuration file or select the default config.")
            return
        if not out_dir.strip():
            st.error("Please provide an output directory.")
            return

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(dir=out_path, prefix="inputs_"))

        receptor_path = save_upload(receptor_upload, temp_dir)
        peptide_path = save_upload(peptide_upload, temp_dir)
        config_path = build_config_path(config_choice, uploaded_config, temp_dir)

        if mode == "Single Run":
            st.write("Running docking pipeline...")
            try:
                raw_cfg = load_config(str(config_path))
                cfg = Config(**raw_cfg)
                result = run_pipeline(cfg, receptor_path, peptide_path, str(out_path))
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)
                return

            st.success("Docking run finished.")
            best_score = result.best_pose.score_cheap
            if best_score is not None:
                st.metric("best_score_cheap", best_score)

            metrics_path = out_path / "metrics.jsonl"
            metrics = parse_metrics(metrics_path)
            if "n_eval" in metrics:
                st.metric("n_eval", metrics["n_eval"])
            if "n_pockets_total" in metrics or "n_pockets_used" in metrics:
                total = metrics.get("n_pockets_total", "-")
                used = metrics.get("n_pockets_used", "-")
                st.write(f"n_pockets_total / n_pockets_used: {total} / {used}")

            series = parse_metrics_series(metrics_path)
            if series:
                st.line_chart(series, x="step", y="score")

            result_path = out_path / "result.json"
            st.write("Artifacts")
            st.code(f"{result_path}")
            st.code(f"{metrics_path}")
        else:
            top_pockets = int(top_pockets_value or 1)
            runs = [
                {
                    "label": "full",
                    "out_dir": out_path / "full",
                    "full_search": True,
                    "top_pockets": None,
                },
                {
                    "label": "reduced",
                    "out_dir": out_path / "reduced",
                    "full_search": False,
                    "top_pockets": top_pockets,
                },
            ]
            results: dict[str, dict[str, Any]] = {}
            st.write("Running docking pipeline (compare mode)...")
            for run in runs:
                run_out_dir = run["out_dir"]
                run_out_dir.mkdir(parents=True, exist_ok=True)
                try:
                    raw_cfg = load_config(str(config_path))
                    cfg = Config(**raw_cfg)
                    cfg.full_search = run["full_search"]
                    if run["top_pockets"] is not None:
                        cfg.top_pockets = run["top_pockets"]
                    start_time = time.perf_counter()
                    result = run_pipeline(cfg, receptor_path, peptide_path, str(run_out_dir))
                    elapsed = time.perf_counter() - start_time
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)
                    return

                metrics_path = run_out_dir / "metrics.jsonl"
                metrics = parse_metrics(metrics_path)
                results[run["label"]] = {
                    "out_dir": str(run_out_dir),
                    "params": {
                        "full_search": run["full_search"],
                        "top_pockets": run["top_pockets"],
                    },
                    "best_score_cheap": result.best_pose.score_cheap,
                    "metrics": metrics,
                    "elapsed_seconds": elapsed,
                }

            st.success("Compare run finished.")
            rows = []
            for label in ("full", "reduced"):
                metrics = results[label]["metrics"]
                rows.append(
                    {
                        "mode": label,
                        "best_score_cheap": results[label]["best_score_cheap"],
                        "n_eval": metrics.get("n_eval"),
                        "n_pockets_total": metrics.get("n_pockets_total"),
                        "n_pockets_used": metrics.get("n_pockets_used"),
                        "reduction_ratio": metrics.get("reduction_ratio"),
                        "elapsed_seconds": results[label]["elapsed_seconds"],
                    }
                )
            st.table(rows)

            for label in ("full", "reduced"):
                metrics_path = Path(results[label]["out_dir"]) / "metrics.jsonl"
                series = parse_metrics_series(metrics_path)
                if series:
                    st.subheader(f"{label} run metrics")
                    st.line_chart(series, x="step", y="score")

            report = {
                "parameters": {
                    "output_root": str(out_path),
                    "top_pockets": top_pockets,
                },
                "full": results["full"],
                "reduced": results["reduced"],
            }
            report_path = out_path / "report.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

            st.write("Artifacts")
            st.code(str(Path(results["full"]["out_dir"]) / "result.json"))
            st.code(str(Path(results["reduced"]["out_dir"]) / "result.json"))
            st.code(str(report_path))
            st.download_button(
                "Download report.json",
                data=report_path.read_text(encoding="utf-8"),
                file_name="report.json",
                mime="application/json",
            )


def main() -> None:
    """Main entrypoint for Streamlit."""

    st.set_page_config(page_title="Docking Reduce", layout="centered")
    init_state()

    if st.session_state.screen == "experiment":
        render_experiment()
    else:
        render_home()


if __name__ == "__main__":
    main()
