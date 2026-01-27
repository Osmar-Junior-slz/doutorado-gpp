"""Docking execution page."""

from __future__ import annotations

import json
import tempfile
import threading
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


def normalize_out_dir(path: str) -> str:
    if not path:
        return ""
    return str(Path(path).expanduser().resolve(strict=False))


def can_browse_for_directory() -> bool:
    try:
        import tkinter  # noqa: F401
        from tkinter import filedialog  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def browse_for_directory() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:  # noqa: BLE001
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        folder = filedialog.askdirectory()
        root.destroy()
    except Exception:  # noqa: BLE001
        return None
    return folder or None


def update_recent_out_dirs(selected: str, max_items: int = 8) -> None:
    if not selected:
        return
    normalized = normalize_out_dir(selected)
    recent = st.session_state.get(StateKeys.RECENT_OUT_DIRS, [])
    if not isinstance(recent, list):
        recent = []
    new_recent = [normalized] + [path for path in recent if path != normalized]
    st.session_state[StateKeys.RECENT_OUT_DIRS] = new_recent[:max_items]


def validate_out_dir(out_dir: str) -> tuple[bool, Path | None, str | None]:
    if not out_dir or not out_dir.strip():
        return False, None, "Informe um diretório de saída."
    normalized = normalize_out_dir(out_dir)
    out_path = Path(normalized)
    try:
        out_path.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        return False, None, "Não foi possível criar o diretório de saída."
    try:
        with tempfile.NamedTemporaryFile(dir=out_path, prefix=".write_test_", delete=True) as handle:
            handle.write(b"ok")
            handle.flush()
    except Exception:  # noqa: BLE001
        return False, None, "Diretório de saída não é gravável."
    return True, out_path, None


def read_last_metrics_step(metrics_path: Path, max_bytes: int = 8192) -> int | None:
    """Lê o metrics.jsonl incremental para recuperar a geração mais recente.

    O pipeline grava uma linha JSON por geração com o campo "step"; usamos a última linha válida
    (ou o maior step encontrado) para estimar o progresso enquanto a execução ainda está em curso.
    """

    try:
        if not metrics_path.exists():
            return None
        with metrics_path.open("rb") as handle:
            try:
                handle.seek(0, 2)
                file_size = handle.tell()
                read_size = min(file_size, max_bytes)
                handle.seek(-read_size, 2)
            except OSError:
                handle.seek(0)
            chunk = handle.read().decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return None

    steps: list[int] = []
    for line in chunk.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = record.get("step")
        if isinstance(step, (int, float)):
            steps.append(int(step))
    return max(steps) if steps else None


def compute_progress(step: int | None, total_generations: int) -> float:
    if total_generations <= 0:
        return 0.0
    current = max(step or 0, 0)
    return min(current / total_generations, 1.0)


def format_progress_text(step: int | None, total_generations: int, progress: float) -> str:
    current = max(step or 0, 0)
    return f"Geração {current} / {total_generations} ({progress * 100:.1f}%)"


def run_pipeline_with_progress(
    cfg: Config,
    receptor_path: str,
    peptide_path: str,
    out_dir: Path,
    total_generations: int,
    progress_bar: st.delta_generator.DeltaGenerator,
    progress_text: st.delta_generator.DeltaGenerator,
    poll_interval: float = 0.3,
) -> tuple[Any, float]:
    result_holder: dict[str, Any] = {}
    metrics_path = out_dir / "metrics.jsonl"

    def _run_pipeline() -> None:
        try:
            start_time = time.perf_counter()
            result_holder["result"] = run_pipeline(cfg, receptor_path, peptide_path, str(out_dir))
            result_holder["elapsed"] = time.perf_counter() - start_time
        except Exception as exc:  # noqa: BLE001
            result_holder["error"] = exc

    thread = threading.Thread(target=_run_pipeline, daemon=True)
    thread.start()

    last_step: int | None = None
    while thread.is_alive():
        last_step = read_last_metrics_step(metrics_path)
        progress_value = compute_progress(last_step, total_generations)
        progress_bar.progress(progress_value)
        progress_text.write(format_progress_text(last_step, total_generations, progress_value))
        time.sleep(poll_interval)

    thread.join()
    last_step = read_last_metrics_step(metrics_path) or last_step
    progress_value = compute_progress(last_step, total_generations)
    if total_generations > 0 and progress_value < 1.0:
        progress_value = 1.0
        last_step = total_generations
    progress_bar.progress(progress_value)
    progress_text.write(format_progress_text(last_step, total_generations, progress_value))

    if "error" in result_holder:
        raise result_holder["error"]
    return result_holder["result"], result_holder.get("elapsed", 0.0)


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
        st.session_state.setdefault(StateKeys.RECENT_OUT_DIRS, [state.default_out_dir])
        st.session_state[StateKeys.OUT_DIR] = normalize_out_dir(st.session_state[StateKeys.OUT_DIR])

        recent_key = "out_dir_recent"
        recent_dirs = st.session_state.get(StateKeys.RECENT_OUT_DIRS, [])
        if not isinstance(recent_dirs, list):
            recent_dirs = []
        current_dir = normalize_out_dir(st.session_state[StateKeys.OUT_DIR])
        default_dir = normalize_out_dir(state.default_out_dir)
        recent_options: list[str] = []
        for candidate in [current_dir, default_dir, *recent_dirs]:
            normalized = normalize_out_dir(candidate)
            if normalized and normalized not in recent_options:
                recent_options.append(normalized)
        if not recent_options:
            recent_options.append(default_dir or current_dir or "runs")

        if recent_key not in st.session_state:
            st.session_state[recent_key] = current_dir or recent_options[0]

        def sync_out_dir_from_recent() -> None:
            selected = st.session_state.get(recent_key, "")
            if selected:
                st.session_state[StateKeys.OUT_DIR] = selected

        def sync_out_dir_from_text() -> None:
            current_value = st.session_state.get(StateKeys.OUT_DIR, "")
            if current_value:
                st.session_state[StateKeys.OUT_DIR] = normalize_out_dir(current_value)

        def sync_out_dir_from_browse() -> None:
            selected = browse_for_directory()
            if selected:
                normalized = normalize_out_dir(selected)
                st.session_state[StateKeys.OUT_DIR] = normalized
                st.session_state[recent_key] = normalized

        st.subheader("Diretório de saída")
        can_browse = can_browse_for_directory()
        col_main, col_action = st.columns([3, 1])
        with col_main:
            selected_index = (
                recent_options.index(st.session_state[recent_key])
                if st.session_state.get(recent_key) in recent_options
                else 0
            )
            st.selectbox(
                "Diretórios recentes",
                options=recent_options,
                index=selected_index,
                key=recent_key,
                on_change=sync_out_dir_from_recent,
            )
        with col_action:
            if can_browse:
                st.button("Procurar...", on_click=sync_out_dir_from_browse)
            else:
                st.button("Procurar...", disabled=True)

        if not can_browse:
            st.warning(
                "Seleção de pastas indisponível neste ambiente. Use os diretórios recentes ou informe o caminho manualmente."
            )

        out_dir = st.text_input(
            "Diretório de saída",
            value=st.session_state[StateKeys.OUT_DIR],
            key=StateKeys.OUT_DIR,
            on_change=sync_out_dir_from_text,
        )

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

        st.subheader("Parâmetros de execução")
        base_cfg = st.session_state.get(StateKeys.LOADED_CONFIG, {})
        current_overrides = dict(st.session_state.get(StateKeys.CONFIG_OVERRIDES, {}))
        resolved_preview = apply_overrides(base_cfg, current_overrides)
        total_generations_preview = int(resolved_preview.get("generations", 0) or 0)
        pop_size_preview = int(resolved_preview.get("pop_size", 0) or 0)

        st.write(f"Total de gerações (configuração resolvida): {total_generations_preview}")
        st.write(f"Tamanho da população (configuração resolvida): {pop_size_preview}")

        st.session_state.setdefault(StateKeys.OVERRIDE_GENERATIONS, total_generations_preview or 1)
        st.session_state.setdefault(StateKeys.OVERRIDE_POP_SIZE, pop_size_preview or 1)

        col_gen, col_pop = st.columns(2)
        with col_gen:
            generations_value = st.number_input(
                "Sobrescrever gerações",
                min_value=1,
                step=1,
                value=int(st.session_state[StateKeys.OVERRIDE_GENERATIONS]),
                key=StateKeys.OVERRIDE_GENERATIONS,
            )
        with col_pop:
            pop_size_value = st.number_input(
                "Sobrescrever pop_size",
                min_value=1,
                step=1,
                value=int(st.session_state[StateKeys.OVERRIDE_POP_SIZE]),
                key=StateKeys.OVERRIDE_POP_SIZE,
            )

        current_overrides["generations"] = int(generations_value)
        current_overrides["pop_size"] = int(pop_size_value)
        st.session_state[StateKeys.CONFIG_OVERRIDES] = current_overrides

        if st.button("Executar"):
            if config_choice == "Enviar YAML" and uploaded_config is None:
                st.error("Envie um arquivo de configuração YAML ou selecione a configuração padrão.")
                return
            valid_out_dir, out_path, error_message = validate_out_dir(out_dir)
            if not valid_out_dir or out_path is None:
                st.error(error_message or "Informe um diretório de saída válido.")
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

            resolved_cfg = apply_overrides(raw_cfg, st.session_state.get(StateKeys.CONFIG_OVERRIDES, {}))
            total_generations = int(resolved_cfg.get("generations", 0) or 0)

            if mode == "Execução única":
                st.write("Executando pipeline de docking...")
                progress_bar = st.progress(0.0)
                progress_text = st.empty()
                try:
                    cfg = Config(**resolved_cfg)
                    result, elapsed = run_pipeline_with_progress(
                        cfg,
                        receptor_path,
                        peptide_path,
                        out_path,
                        total_generations,
                        progress_bar,
                        progress_text,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)
                    return

                write_resolved_config(out_path, resolved_cfg)
                set_state(**{StateKeys.LAST_OUT_DIR: str(out_path)})
                update_recent_out_dirs(str(out_path))

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
            progress_container = st.container()
            with progress_container:
                st.subheader("Progresso da comparação")
                col_full, col_reduced = st.columns(2)
                with col_full:
                    st.markdown("**Completo**")
                    progress_full = st.progress(0.0)
                    progress_full_text = st.empty()
                with col_reduced:
                    st.markdown("**Reduzido**")
                    progress_reduced = st.progress(0.0)
                    progress_reduced_text = st.empty()
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
                    progress_bar = progress_full if run["label"] == "full" else progress_reduced
                    progress_text = progress_full_text if run["label"] == "full" else progress_reduced_text
                    result, elapsed = run_pipeline_with_progress(
                        cfg,
                        receptor_path,
                        peptide_path,
                        run_out_dir,
                        total_generations,
                        progress_bar,
                        progress_text,
                    )
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
            update_recent_out_dirs(str(out_path))

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
