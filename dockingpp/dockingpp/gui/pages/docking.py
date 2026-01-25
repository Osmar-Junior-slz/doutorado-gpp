"""Docking execution page."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from dockingpp.data.io import load_config
from dockingpp.pipeline.run import Config, run_pipeline
from dockingpp.gui.pages.base import BasePage
from dockingpp.gui.services.io_service import ensure_dir, save_uploaded_file
from dockingpp.gui.services.report_service import load_jsonl, metrics_series, summarize_metrics
from dockingpp.gui.state import AppState, DEFAULT_CONFIG_PATH, StateKeys, set_state
from dockingpp.gui.ui.components import download_json_button


def write_resolved_config(out_dir: Path, cfg_dict: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.resolved.json"
    path.write_text(json.dumps(cfg_dict, indent=2, ensure_ascii=False), encoding="utf-8")


def apply_overrides(raw_cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(raw_cfg)
    for key, value in overrides.items():
        if value is None:
            continue
        resolved[key] = value
    return resolved


class DockingPage(BasePage):
    id = "Docking"
    title = "Docking"

    def render(self, state: AppState) -> None:
        st.header("Execução de Docking")

        prepared_path = st.session_state.get(StateKeys.PREPARED_RECEPTOR_PATH, "")
        use_prepared = False
        if prepared_path:
            st.info(f"Receptor preparado disponível: {prepared_path}")
            use_prepared = st.checkbox("Usar receptor preparado", value=True, key="use_prepared_receptor")

        receptor_upload = st.file_uploader("Receptor (.pdb)", type=["pdb"], key="receptor_upload")
        peptide_upload = st.file_uploader("Peptídeo (.pdb)", type=["pdb"], key="peptide_upload")

        st.session_state.setdefault(StateKeys.OUT_DIR, state.default_out_dir)
        out_dir = st.text_input("Diretório de saída", value=st.session_state[StateKeys.OUT_DIR], key=StateKeys.OUT_DIR)

        st.subheader("Modo de experimento")
        mode = st.radio(
            "Selecione o modo",
            options=["Execução única", "Comparar (Completo vs Reduzido)"],
            index=0,
            key=StateKeys.RUN_MODE,
        )
        top_pockets_value = None
        if mode == "Comparar (Completo vs Reduzido)":
            top_pockets_value = st.slider(
                "Melhores bolsões (execução reduzida)",
                min_value=1,
                max_value=20,
                value=5,
                key=StateKeys.COMPARE_TOP_POCKETS,
            )

        st.subheader("Configuração")
        config_choice = st.radio(
            "Configuração",
            options=["Padrão (configs/default.yaml)", "Enviar YAML"],
            index=0,
            key=StateKeys.CONFIG_CHOICE,
        )
        uploaded_config = None
        if config_choice == "Enviar YAML":
            uploaded_config = st.file_uploader("Configuração YAML", type=["yaml", "yml"], key=StateKeys.CONFIG_UPLOAD)
            if uploaded_config is not None:
                try:
                    config_text = uploaded_config.getvalue().decode("utf-8")
                    set_state(
                        **{
                            StateKeys.LOADED_CONFIG: yaml.safe_load(config_text) or {},
                            StateKeys.CONFIG_SOURCE_LABEL: f"YAML enviado: {uploaded_config.name}",
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error("Não foi possível ler o YAML enviado.")
                    st.exception(exc)
        else:
            set_state(
                **{
                    StateKeys.LOADED_CONFIG: load_config(str(DEFAULT_CONFIG_PATH)),
                    StateKeys.CONFIG_SOURCE_LABEL: "Padrão (configs/default.yaml)",
                }
            )

        st.caption("A configuração pode ser ajustada na aba Configurações.")

        if st.button("Executar"):
            if config_choice == "Enviar YAML" and uploaded_config is None:
                st.error("Envie um arquivo de configuração YAML ou selecione a configuração padrão.")
                return
            if not out_dir.strip():
                st.error("Informe um diretório de saída.")
                return
            if use_prepared:
                prepared_file = Path(prepared_path)
                if not prepared_file.exists():
                    st.error(
                        "Arquivo do receptor preparado não foi encontrado no disco: "
                        f"{prepared_file}. Gere novamente ou faça upload."
                    )
                    return
            else:
                if receptor_upload is None:
                    st.error("Envie os arquivos PDB de receptor e peptídeo antes de executar.")
                    return
            if peptide_upload is None:
                st.error("Envie os arquivos PDB de receptor e peptídeo antes de executar.")
                return

            out_path = Path(out_dir)
            ensure_dir(out_path)
            inputs_dir = ensure_dir(out_path / "inputs")

            if use_prepared:
                receptor_path = str(prepared_file)
            else:
                receptor_path = str(save_uploaded_file(receptor_upload, inputs_dir, filename_hint="receptor.pdb"))
            peptide_path = str(save_uploaded_file(peptide_upload, inputs_dir, filename_hint="peptide.pdb"))

            if config_choice == "Enviar YAML" and uploaded_config is not None:
                config_path = save_uploaded_file(uploaded_config, inputs_dir)
                raw_cfg = load_config(str(config_path))
            else:
                raw_cfg = load_config(str(DEFAULT_CONFIG_PATH))

            resolved_cfg = apply_overrides(raw_cfg, state.config_overrides)

            if mode == "Execução única":
                st.write("Executando pipeline de docking...")
                start_time = time.perf_counter()
                try:
                    cfg = Config(**resolved_cfg)
                    result = run_pipeline(cfg, receptor_path, peptide_path, str(out_path))
                    elapsed = time.perf_counter() - start_time
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)
                    return

                write_resolved_config(out_path, resolved_cfg)
                set_state(**{StateKeys.LAST_OUT_DIR: str(out_path)})

                st.success("Execução de docking concluída.")
                metrics_path = out_path / "metrics.jsonl"
                metrics_records = load_jsonl(metrics_path)
                metrics = summarize_metrics(metrics_records)
                series, _ = metrics_series(metrics_records, ["best_score_cheap", "best_score"])
                if series:
                    st.line_chart(series, x="step", y="score")

                st.subheader("Resumo da execução")
                summary_rows = [
                    {"Campo": "Melhor score (cheap)", "Valor": result.best_pose.score_cheap},
                    {"Campo": "Avaliações", "Valor": metrics.get("n_eval")},
                    {"Campo": "Bolsões usados", "Valor": metrics.get("n_pockets_used")},
                    {"Campo": "Razão de redução", "Valor": metrics.get("reduction_ratio")},
                    {"Campo": "Tempo (s)", "Valor": round(elapsed, 2)},
                ]
                st.table(summary_rows)

                st.write("Artefatos")
                result_path = out_path / "result.json"
                download_json_button("Baixar result.json", result_path, filename="result.json", warn_missing=True)
                return

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
            resolved_configs: dict[str, dict[str, Any]] = {}
            st.write("Executando pipeline de docking (modo de comparação)...")
            for run in runs:
                run_out_dir = run["out_dir"]
                run_out_dir.mkdir(parents=True, exist_ok=True)
                run_cfg_dict = dict(resolved_cfg)
                run_cfg_dict["full_search"] = run["full_search"]
                if run["top_pockets"] is not None:
                    run_cfg_dict["top_pockets"] = run["top_pockets"]
                resolved_configs[run["label"]] = run_cfg_dict
                try:
                    cfg = Config(**run_cfg_dict)
                    start_time = time.perf_counter()
                    result = run_pipeline(cfg, receptor_path, peptide_path, str(run_out_dir))
                    elapsed = time.perf_counter() - start_time
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)
                    return

                write_resolved_config(run_out_dir, run_cfg_dict)

                metrics_path = run_out_dir / "metrics.jsonl"
                metrics = summarize_metrics(load_jsonl(metrics_path))
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

            st.success("Execução de comparação concluída.")
            set_state(**{StateKeys.LAST_OUT_DIR: str(out_path)})

            rows = []
            for label in ("full", "reduced"):
                metrics = results[label]["metrics"]
                label_name = "Completo" if label == "full" else "Reduzido"
                rows.append(
                    {
                        "Modo": label_name,
                        "Melhor score (cheap)": results[label]["best_score_cheap"],
                        "Avaliações": metrics.get("n_eval"),
                        "Bolsões totais": metrics.get("n_pockets_total"),
                        "Bolsões usados": metrics.get("n_pockets_used"),
                        "Razão de redução": metrics.get("reduction_ratio"),
                        "Tempo (s)": results[label]["elapsed_seconds"],
                    }
                )
            st.table(rows)

            for label in ("full", "reduced"):
                metrics_path = Path(results[label]["out_dir"]) / "metrics.jsonl"
                series, _ = metrics_series(load_jsonl(metrics_path), ["best_score_cheap", "best_score"])
                if series:
                    label_name = "Completo" if label == "full" else "Reduzido"
                    st.subheader(f"Métricas da execução {label_name.lower()}")
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
            write_resolved_config(out_path, {"full": resolved_configs["full"], "reduced": resolved_configs["reduced"]})

            st.subheader("Resumo da execução")
            st.table(rows)

            st.write("Artefatos")
            result_full = Path(results["full"]["out_dir"]) / "result.json"
            result_reduced = Path(results["reduced"]["out_dir"]) / "result.json"
            download_json_button("Baixar report.json", report_path, filename="report.json", warn_missing=True)
            download_json_button(
                "Baixar result.json (completo)",
                result_full,
                filename="result_full.json",
                warn_missing=False,
            )
            download_json_button(
                "Baixar result.json (reduzido)",
                result_reduced,
                filename="result_reduced.json",
                warn_missing=False,
            )
