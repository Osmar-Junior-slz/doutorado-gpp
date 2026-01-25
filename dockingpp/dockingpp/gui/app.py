"""Streamlit GUI for dockingpp."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import streamlit as st
import yaml

from dockingpp.data.io import load_config
from dockingpp.data.pdb_clean import clean_pdb_text
from dockingpp.pipeline.run import Config, run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def init_state() -> None:
    """Initialize Streamlit session state."""

    if "page" not in st.session_state:
        st.session_state.page = "Início"
    if "default_out_dir" not in st.session_state:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.default_out_dir = f"runs/run_{timestamp}"
    if "loaded_config" not in st.session_state:
        st.session_state.loaded_config = load_config(str(DEFAULT_CONFIG_PATH))
    if "config_source_label" not in st.session_state:
        st.session_state.config_source_label = "Padrão (configs/default.yaml)"
    if "config_overrides" not in st.session_state:
        st.session_state.config_overrides = {}
    if "last_out_dir" not in st.session_state:
        st.session_state.last_out_dir = ""
    if "prepared_receptor_path" not in st.session_state:
        st.session_state.prepared_receptor_path = ""
    if "prepared_pdb_text" not in st.session_state:
        st.session_state.prepared_pdb_text = ""
    if "prepared_pdb_name" not in st.session_state:
        st.session_state.prepared_pdb_name = ""
    if "pdb_clean_outdir" not in st.session_state:
        st.session_state.pdb_clean_outdir = "datasets/cleaned"


def save_upload(upload: st.runtime.uploaded_file_manager.UploadedFile, dest_dir: Path) -> str:
    """Persist an uploaded file to disk and return its path."""

    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / upload.name
    with open(path, "wb") as handle:
        handle.write(upload.getbuffer())
    return str(path)


def parse_metrics(path: Path) -> dict[str, Any]:
    """Parse metrics JSONL into a summary dictionary."""

    summary: dict[str, Any] = {}
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


def parse_metrics_series(path: Path, keys: list[str]) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Parse metrics JSONL into a list of points for charting."""

    if not path.exists():
        return [], None

    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(record)

    selected_key = next((key for key in keys if any(rec.get("name") == key for rec in records)), None)
    if not selected_key:
        return [], None

    series: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        if record.get("name") != selected_key:
            continue
        step = record.get("step")
        if step is None:
            step = record.get("generation")
        if step is None:
            step = idx
        series.append({"step": step, "score": record.get("value")})
    return series, selected_key


def write_resolved_config(out_dir: Path, cfg_dict: dict[str, Any]) -> None:
    """Persist the resolved configuration used for execution."""

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.resolved.json"
    path.write_text(json.dumps(cfg_dict, indent=2, ensure_ascii=False), encoding="utf-8")


def apply_overrides(raw_cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply in-memory overrides to a configuration dictionary."""

    resolved = dict(raw_cfg)
    for key, value in overrides.items():
        if value is None:
            continue
        resolved[key] = value
    return resolved


def render_home() -> None:
    """Render the landing screen."""

    st.header("Docking Reduce")
    st.write("Interface local para executar experimentos de docking e comparar busca completa vs reduzida.")
    if st.button("Novo experimento"):
        st.session_state.page = "Docking"
        st.rerun()


def render_docking() -> None:
    """Render the docking screen."""

    st.header("Execução de Docking")

    prepared_path = st.session_state.get("prepared_receptor_path", "")
    use_prepared = False
    if prepared_path:
        st.info(f"Receptor preparado disponível: {prepared_path}")
        use_prepared = st.checkbox("Usar receptor preparado", value=True, key="use_prepared_receptor")

    receptor_upload = st.file_uploader("Receptor (.pdb)", type=["pdb"], key="receptor_upload")
    peptide_upload = st.file_uploader("Peptídeo (.pdb)", type=["pdb"], key="peptide_upload")

    out_dir = st.text_input("Diretório de saída", value=st.session_state.default_out_dir, key="out_dir")

    st.subheader("Modo de experimento")
    mode = st.radio(
        "Selecione o modo",
        options=["Execução única", "Comparar (Completo vs Reduzido)"],
        index=0,
        key="run_mode",
    )
    top_pockets_value = None
    if mode == "Comparar (Completo vs Reduzido)":
        top_pockets_value = st.slider(
            "Melhores bolsões (execução reduzida)",
            min_value=1,
            max_value=20,
            value=5,
            key="compare_top_pockets",
        )

    st.subheader("Configuração")
    config_choice = st.radio(
        "Configuração",
        options=["Padrão (configs/default.yaml)", "Enviar YAML"],
        index=0,
        key="config_choice",
    )
    uploaded_config = None
    if config_choice == "Enviar YAML":
        uploaded_config = st.file_uploader("Configuração YAML", type=["yaml", "yml"], key="config_upload")
        if uploaded_config is not None:
            try:
                config_text = uploaded_config.getvalue().decode("utf-8")
                st.session_state.loaded_config = yaml.safe_load(config_text) or {}
                st.session_state.config_source_label = f"YAML enviado: {uploaded_config.name}"
            except Exception as exc:  # noqa: BLE001
                st.error("Não foi possível ler o YAML enviado.")
                st.exception(exc)
    else:
        st.session_state.loaded_config = load_config(str(DEFAULT_CONFIG_PATH))
        st.session_state.config_source_label = "Padrão (configs/default.yaml)"

    st.caption("A configuração pode ser ajustada na aba Configurações.")

    if st.button("Executar"):
        if (receptor_upload is None and not use_prepared) or peptide_upload is None:
            st.error("Envie os arquivos PDB de receptor e peptídeo antes de executar.")
            return
        if config_choice == "Enviar YAML" and uploaded_config is None:
            st.error("Envie um arquivo de configuração YAML ou selecione a configuração padrão.")
            return
        if not out_dir.strip():
            st.error("Informe um diretório de saída.")
            return

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(dir=out_path, prefix="inputs_"))

        if use_prepared:
            prepared_file = Path(prepared_path)
            if not prepared_file.exists():
                st.error("O receptor preparado não está mais disponível. Faça o upload novamente.")
                return
            receptor_path = str(prepared_file)
        else:
            receptor_path = save_upload(receptor_upload, temp_dir)
        peptide_path = save_upload(peptide_upload, temp_dir)

        if config_choice == "Enviar YAML" and uploaded_config is not None:
            config_path = Path(save_upload(uploaded_config, temp_dir))
            raw_cfg = load_config(str(config_path))
        else:
            config_path = DEFAULT_CONFIG_PATH
            raw_cfg = load_config(str(config_path))

        overrides = st.session_state.get("config_overrides", {})
        resolved_cfg = apply_overrides(raw_cfg, overrides)

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
            st.session_state.last_out_dir = str(out_path)

            st.success("Execução de docking concluída.")
            metrics_path = out_path / "metrics.jsonl"
            metrics = parse_metrics(metrics_path)
            series, _ = parse_metrics_series(metrics_path, ["best_score_cheap", "best_score"])
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

            result_path = out_path / "result.json"
            st.write("Artefatos")
            if result_path.exists():
                st.download_button(
                    "Baixar result.json",
                    data=result_path.read_text(encoding="utf-8"),
                    file_name="result.json",
                    mime="application/json",
                )
            else:
                st.warning("result.json não encontrado na execução.")
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
            st.write("Executando pipeline de docking (modo de comparação)...")
            resolved_configs: dict[str, dict[str, Any]] = {}
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

            st.success("Execução de comparação concluída.")
            st.session_state.last_out_dir = str(out_path)

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
                series, _ = parse_metrics_series(metrics_path, ["best_score_cheap", "best_score"])
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
            if report_path.exists():
                st.download_button(
                    "Baixar report.json",
                    data=report_path.read_text(encoding="utf-8"),
                    file_name="report.json",
                    mime="application/json",
                )
            if result_full.exists():
                st.download_button(
                    "Baixar result.json (completo)",
                    data=result_full.read_text(encoding="utf-8"),
                    file_name="result_full.json",
                    mime="application/json",
                )
            if result_reduced.exists():
                st.download_button(
                    "Baixar result.json (reduzido)",
                    data=result_reduced.read_text(encoding="utf-8"),
                    file_name="result_reduced.json",
                    mime="application/json",
                )


def render_config() -> None:
    """Render the configuration screen."""

    st.header("Configurações")
    config_data = st.session_state.get("loaded_config", {})
    source_label = st.session_state.get("config_source_label", "Padrão (configs/default.yaml)")

    st.write(f"Configuração carregada: {source_label}")
    if config_data:
        st.code(yaml.safe_dump(config_data, sort_keys=False), language="yaml")
    else:
        st.warning("Nenhuma configuração carregada no momento.")

    st.subheader("Sobrescrever parâmetros")
    st.write("As alterações abaixo são aplicadas apenas na execução atual (em memória).")

    seed = st.number_input(
        "seed",
        value=int(st.session_state.get("override_seed", config_data.get("seed", 7))),
        step=1,
        key="override_seed",
    )
    generations = st.number_input(
        "generations",
        value=int(st.session_state.get("override_generations", config_data.get("generations", 5))),
        step=1,
        key="override_generations",
    )
    pop_size = st.number_input(
        "pop_size",
        value=int(st.session_state.get("override_pop_size", config_data.get("pop_size", 20))),
        step=1,
        key="override_pop_size",
    )
    topk = st.number_input(
        "topk",
        value=int(st.session_state.get("override_topk", config_data.get("topk", 5))),
        step=1,
        key="override_topk",
    )
    full_search = st.checkbox(
        "full_search",
        value=bool(st.session_state.get("override_full_search", config_data.get("full_search", True))),
        key="override_full_search",
    )

    top_pockets = None
    if not full_search:
        top_pockets = st.number_input(
            "top_pockets",
            value=int(st.session_state.get("override_top_pockets", config_data.get("top_pockets", 3))),
            step=1,
            key="override_top_pockets",
        )

    st.session_state.config_overrides = {
        "seed": int(seed),
        "generations": int(generations),
        "pop_size": int(pop_size),
        "topk": int(topk),
        "full_search": bool(full_search),
        "top_pockets": int(top_pockets) if top_pockets is not None else None,
    }


def render_reports() -> None:
    """Render the reports screen."""

    st.header("Relatórios")
    default_root = st.session_state.get("reports_root", "runs")
    root_input = st.text_input("Diretório de execuções", value=default_root)

    if st.session_state.get("last_out_dir"):
        st.caption(f"Última execução: {st.session_state.last_out_dir}")

    if st.button("Buscar execuções"):
        base_dir = Path(root_input).expanduser()
        if not base_dir.exists():
            st.warning("Diretório informado não existe.")
            st.session_state.report_runs = []
        else:
            runs = [
                child
                for child in base_dir.iterdir()
                if child.is_dir()
                and ((child / "result.json").exists() or (child / "report.json").exists())
            ]
            st.session_state.report_runs = sorted(runs)
            st.session_state.reports_root = root_input

    runs = st.session_state.get("report_runs", [])
    if not runs:
        st.info("Nenhuma execução encontrada ainda. Clique em 'Buscar execuções'.")
        return

    base_dir = Path(st.session_state.get("reports_root", "runs")).expanduser()
    options = [str(run.relative_to(base_dir)) if run.is_relative_to(base_dir) else str(run) for run in runs]
    default_index = 0
    last_out_dir = st.session_state.get("last_out_dir")
    if last_out_dir:
        try:
            last_path = Path(last_out_dir)
            if last_path in runs:
                default_index = runs.index(last_path)
        except Exception:  # noqa: BLE001
            default_index = 0

    selected = st.selectbox("Execução", options=options, index=default_index)
    selected_path = runs[options.index(selected)]

    report_path = selected_path / "report.json"
    result_path = selected_path / "result.json"
    metrics_path = selected_path / "metrics.jsonl"

    if report_path.exists():
        try:
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
            rows = []
            for label in ("full", "reduced"):
                if label not in report_data:
                    continue
                metrics = report_data[label].get("metrics", {})
                label_name = "Completo" if label == "full" else "Reduzido"
                rows.append(
                    {
                        "Modo": label_name,
                        "Melhor score (cheap)": report_data[label].get("best_score_cheap"),
                        "Avaliações": metrics.get("n_eval"),
                        "Bolsões totais": metrics.get("n_pockets_total"),
                        "Bolsões usados": metrics.get("n_pockets_used"),
                        "Razão de redução": metrics.get("reduction_ratio"),
                        "Tempo (s)": report_data[label].get("elapsed_seconds"),
                    }
                )
            if rows:
                st.subheader("Comparação Full vs Reduced")
                st.table(rows)
            else:
                st.warning("report.json encontrado, mas sem dados de comparação completos.")
        except Exception as exc:  # noqa: BLE001
            st.warning("Não foi possível ler report.json.")
            st.exception(exc)
    else:
        st.warning("report.json não encontrado nesta execução.")

    metrics_summary = parse_metrics(metrics_path)
    if result_path.exists():
        try:
            result_data = json.loads(result_path.read_text(encoding="utf-8"))
            st.subheader("Resumo da execução")
            summary_rows = [
                {"Campo": "Melhor score (cheap)", "Valor": result_data.get("best_score_cheap")},
                {"Campo": "Avaliações", "Valor": metrics_summary.get("n_eval")},
                {"Campo": "Bolsões usados", "Valor": metrics_summary.get("n_pockets_used")},
                {"Campo": "Razão de redução", "Valor": metrics_summary.get("reduction_ratio")},
                {"Campo": "Tempo (s)", "Valor": result_data.get("elapsed_s")},
            ]
            st.table(summary_rows)
        except Exception as exc:  # noqa: BLE001
            st.warning("Não foi possível ler result.json.")
            st.exception(exc)
    else:
        st.warning("result.json não encontrado nesta execução.")

    if metrics_path.exists():
        series, series_key = parse_metrics_series(metrics_path, ["best_score_cheap", "best_score", "best"])
        if series:
            st.subheader("Evolução do score")
            st.line_chart(series, x="step", y="score")
        else:
            st.warning("metrics.jsonl encontrado, mas não há dados de score para plotar.")
    else:
        st.warning("metrics.jsonl não encontrado nesta execução.")

    st.write("Downloads")
    if result_path.exists():
        st.download_button(
            "Baixar result.json",
            data=result_path.read_text(encoding="utf-8"),
            file_name="result.json",
            mime="application/json",
        )
    if report_path.exists():
        st.download_button(
            "Baixar report.json",
            data=report_path.read_text(encoding="utf-8"),
            file_name="report.json",
            mime="application/json",
        )
    if metrics_path.exists():
        st.download_button(
            "Baixar metrics.jsonl",
            data=metrics_path.read_text(encoding="utf-8"),
            file_name="metrics.jsonl",
            mime="application/json",
        )


def parse_keep_list(value: str) -> set[str]:
    return {item.strip().upper() for item in value.split(",") if item.strip()}


def compute_pdb_counts(lines: list[str]) -> dict[str, int]:
    total = len(lines)
    n_atom = 0
    n_hetatm = 0
    for line in lines:
        record = line[:6].strip().upper()
        if record == "ATOM":
            n_atom += 1
        elif record == "HETATM":
            n_hetatm += 1
    return {"total": total, "atom": n_atom, "hetatm": n_hetatm}


def choose_directory_dialog() -> str | None:
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


def render_pdb_prep() -> None:
    """Render the PDB preparation screen."""

    st.header("Preparação PDB")
    upload = st.file_uploader("Arquivo PDB", type=["pdb"], key="pdb_prep_upload")

    remove_waters = st.checkbox("Remover águas", value=True, key="pdb_remove_waters")
    remove_hetatm = st.checkbox("Remover HETATM", value=True, key="pdb_remove_hetatm")
    remove_ions = st.checkbox("Remover íons", value=True, key="pdb_remove_ions")
    keep_list_raw = st.text_input("Manter resíduos HETATM (ex.: HEM,ZN)", value="", key="pdb_keep_list")
    output_dir = st.text_input("Pasta de saída", value=st.session_state.pdb_clean_outdir, key="pdb_clean_outdir")
    if st.button("Selecionar pasta..."):
        selected_dir = choose_directory_dialog()
        if selected_dir:
            st.session_state.pdb_clean_outdir = selected_dir
            output_dir = selected_dir
        else:
            st.info("Seleção por diálogo indisponível; digite o caminho manualmente.")

    cleaned_text = st.session_state.get("prepared_pdb_text", "")
    cleaned_name = st.session_state.get("prepared_pdb_name", "cleaned.pdb")

    if st.button("Limpar e salvar"):
        if upload is None:
            st.error("Envie um arquivo PDB para continuar.")
            return
        if not output_dir.strip():
            st.error("Informe uma pasta de saída.")
            return
        pdb_text = upload.getvalue().decode("utf-8")
        keep_set = parse_keep_list(keep_list_raw)
        cleaned_text = clean_pdb_text(
            pdb_text,
            remove_waters=remove_waters,
            remove_hetatm=remove_hetatm,
            remove_ions=remove_ions,
            keep_het_resnames=keep_set or None,
        )
        output_path = Path(output_dir).expanduser()
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            st.error("Sem permissão para criar a pasta de saída.")
            return
        except OSError as exc:
            st.error(f"Não foi possível criar a pasta de saída: {exc}")
            return
        cleaned_name = f"{Path(upload.name).stem}.cleaned.pdb"
        cleaned_file = output_path / cleaned_name
        cleaned_file.write_text(cleaned_text, encoding="utf-8")

        original_lines = [line.rstrip("\n") for line in pdb_text.splitlines()]
        cleaned_lines = [line.rstrip("\n") for line in cleaned_text.splitlines()]
        original_counts = compute_pdb_counts(original_lines)
        cleaned_counts = compute_pdb_counts(cleaned_lines)
        removed = max(original_counts["total"] - cleaned_counts["total"], 0)

        st.success(f"Arquivo salvo em: {cleaned_file}")
        st.write("Resumo da limpeza:")
        st.write(
            {
                "Linhas totais": original_counts["total"],
                "ATOM": original_counts["atom"],
                "HETATM": original_counts["hetatm"],
                "Removidas": removed,
            }
        )

        st.session_state.prepared_receptor_path = str(cleaned_file)
        st.session_state.prepared_pdb_text = cleaned_text
        st.session_state.prepared_pdb_name = cleaned_name

    if cleaned_text:
        download_path = Path(st.session_state.prepared_receptor_path)
        st.download_button(
            "Baixar PDB limpo",
            data=download_path.read_bytes() if download_path.exists() else cleaned_text.encode("utf-8"),
            file_name=cleaned_name,
            mime="chemical/x-pdb",
        )
        if st.button("Usar este receptor no Docking"):
            st.success("Receptor preparado selecionado para o Docking.")
            st.session_state.page = "Docking"
            st.rerun()


def main() -> None:
    """Main entrypoint for Streamlit."""

    st.set_page_config(page_title="Docking Reduce", layout="centered")
    init_state()

    st.sidebar.title("Docking Reduce")
    pages = ["Início", "Docking", "Preparação PDB", "Configurações", "Relatórios"]
    current_page = st.session_state.page if st.session_state.page in pages else "Início"
    selection = st.sidebar.radio("Menu", options=pages, index=pages.index(current_page))
    if selection != st.session_state.page:
        st.session_state.page = selection

    if st.session_state.page == "Início":
        render_home()
    elif st.session_state.page == "Docking":
        render_docking()
    elif st.session_state.page == "Preparação PDB":
        render_pdb_prep()
    elif st.session_state.page == "Configurações":
        render_config()
    else:
        render_reports()


if __name__ == "__main__":
    main()
