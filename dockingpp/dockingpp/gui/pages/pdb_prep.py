"""PDB preparation page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dockingpp.data.pdb_clean import clean_pdb_text
from dockingpp.data.pdb_extract import extract_ligands
from dockingpp.gui.pages.base import BasePage
from dockingpp.gui.services.dialog_service import choose_directory
from dockingpp.gui.state import AppState, StateKeys, set_state


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


def count_ligand_atoms(pdb_text: str) -> int:
    count = 0
    for line in pdb_text.splitlines():
        record = line[:6].strip().upper()
        if record in {"ATOM", "HETATM"}:
            count += 1
    return count


class PdbPrepPage(BasePage):
    id = "Preparação PDB"
    title = "Preparação PDB"

    def render(self, state: AppState) -> None:
        st.header("Preparação PDB")
        upload = st.file_uploader("Arquivo PDB", type=["pdb"], key="pdb_prep_upload")

        remove_waters = st.checkbox("Remover águas", value=True, key="pdb_remove_waters")
        remove_hetatm = st.checkbox("Remover HETATM", value=True, key="pdb_remove_hetatm")
        remove_ions = st.checkbox("Remover íons", value=True, key="pdb_remove_ions")
        keep_list_raw = st.text_input("Manter resíduos HETATM (ex.: HEM,ZN)", value="", key="pdb_keep_list")

        st.session_state.setdefault(StateKeys.PDB_CLEAN_OUTDIR, state.pdb_prep_outdir)
        if st.session_state.get(StateKeys.PDB_CLEAN_OUTDIR_SELECTED):
            st.session_state[StateKeys.PDB_CLEAN_OUTDIR] = st.session_state.pop(StateKeys.PDB_CLEAN_OUTDIR_SELECTED)
        output_dir = st.text_input(
            "Pasta de saída",
            value=st.session_state[StateKeys.PDB_CLEAN_OUTDIR],
            key=StateKeys.PDB_CLEAN_OUTDIR,
        )
        if st.button("Selecionar pasta..."):
            selected_dir = choose_directory()
            if selected_dir:
                st.session_state[StateKeys.PDB_CLEAN_OUTDIR_SELECTED] = selected_dir
                st.rerun()
            else:
                st.info("Seleção por diálogo indisponível; digite o caminho manualmente.")

        cleaned_text = st.session_state.get(StateKeys.PREPARED_PDB_TEXT, "")
        cleaned_name = st.session_state.get(StateKeys.PREPARED_PDB_NAME, "cleaned.pdb")

        st.subheader("Extração de ligantes")
        include_waters = st.checkbox("Incluir águas na extração", value=False, key="pdb_extract_waters")
        include_ions = st.checkbox("Incluir íons na extração", value=False, key="pdb_extract_ions")

        if st.button("Detectar ligantes"):
            if upload is None:
                st.error("Envie um arquivo PDB para detectar ligantes.")
            else:
                pdb_text = upload.getvalue().decode("utf-8")
                ligands = extract_ligands(
                    pdb_text,
                    include_waters=include_waters,
                    include_ions=include_ions,
                )
                set_state(
                    **{
                        StateKeys.PDB_EXTRACT_LIGANDS: ligands,
                        StateKeys.PDB_EXTRACT_FILES: {},
                    }
                )

        ligands: dict[str, str] = st.session_state.get(StateKeys.PDB_EXTRACT_LIGANDS, {})
        if ligands:
            ligand_rows = [
                {"Ligante": ligand_id, "Linhas": count_ligand_atoms(pdb_text)}
                for ligand_id, pdb_text in ligands.items()
            ]
            st.table(ligand_rows)
            selected_ligands = st.multiselect(
                "Selecionar ligantes para extrair",
                options=list(ligands.keys()),
                default=list(ligands.keys()),
                key="pdb_extract_selected",
            )
            if st.button("Extrair selecionados"):
                if not output_dir.strip():
                    st.error("Informe uma pasta de saída.")
                elif not selected_ligands:
                    st.info("Selecione ao menos um ligante para extrair.")
                else:
                    output_path = Path(output_dir).expanduser()
                    try:
                        output_path.mkdir(parents=True, exist_ok=True)
                    except PermissionError:
                        st.error("Sem permissão para criar a pasta de saída.")
                        return
                    except OSError as exc:
                        st.error(f"Não foi possível criar a pasta de saída: {exc}")
                        return
                    extracted_files: dict[str, Path] = {}
                    for ligand_id in selected_ligands:
                        ligand_text = ligands.get(ligand_id)
                        if not ligand_text:
                            continue
                        filename = f"ligand_{ligand_id}.pdb"
                        file_path = output_path / filename
                        try:
                            file_path.write_text(ligand_text, encoding="utf-8")
                        except OSError as exc:
                            st.error(f"Erro ao salvar {filename}: {exc}")
                            continue
                        extracted_files[ligand_id] = file_path
                    set_state(**{StateKeys.PDB_EXTRACT_FILES: extracted_files})
                    if extracted_files:
                        st.success("Ligantes extraídos com sucesso.")
        elif st.session_state.get(StateKeys.PDB_EXTRACT_LIGANDS) is not None:
            st.info("Nenhum ligante encontrado (HETATM) com os filtros atuais.")

        extracted_files = st.session_state.get(StateKeys.PDB_EXTRACT_FILES, {})
        if extracted_files:
            st.write("Downloads")
            for ligand_id, file_path in extracted_files.items():
                ligand_text = ligands.get(ligand_id, "")
                data = file_path.read_bytes() if file_path.exists() else ligand_text.encode("utf-8")
                st.download_button(
                    f"Baixar {file_path.name}",
                    data=data,
                    file_name=file_path.name,
                    mime="chemical/x-pdb",
                )

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

            set_state(
                **{
                    StateKeys.PREPARED_RECEPTOR_PATH: str(cleaned_file),
                    StateKeys.PREPARED_PDB_TEXT: cleaned_text,
                    StateKeys.PREPARED_PDB_NAME: cleaned_name,
                }
            )

        if cleaned_text:
            download_path = Path(st.session_state[StateKeys.PREPARED_RECEPTOR_PATH])
            st.download_button(
                "Baixar PDB limpo",
                data=download_path.read_bytes() if download_path.exists() else cleaned_text.encode("utf-8"),
                file_name=cleaned_name,
                mime="chemical/x-pdb",
            )
            if st.button("Usar este receptor no Docking"):
                st.success("Receptor preparado selecionado para o Docking.")
                set_state(**{StateKeys.PAGE: "Docking"})
                st.rerun()

