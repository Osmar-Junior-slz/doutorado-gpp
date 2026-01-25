"""Utilities for extracting ligands from PDB text."""

from __future__ import annotations

from typing import Iterable

from dockingpp.data.pdb_clean import ION_RESNAMES, WATER_RESNAMES

PROTEIN_RESNAMES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "SEC",
    "PYL",
}


def _parse_ligand_id(line: str) -> str:
    resname = line[17:20].strip()
    chain_id = line[21].strip()
    resseq = line[22:26].strip()
    icode = line[26].strip()
    return f"{resname}_{chain_id or 'X'}_{resseq}{icode or ''}"


def _should_include_resname(resname: str, include_waters: bool, include_ions: bool) -> bool:
    resname_upper = resname.upper()
    if not include_waters and resname_upper in WATER_RESNAMES:
        return False
    if not include_ions and resname_upper in ION_RESNAMES:
        return False
    return True


def _iter_model_lines(lines: Iterable[str]) -> tuple[list[str], str | None, str | None]:
    selected: list[str] = []
    model_header = None
    model_end = None
    found_model = False
    in_model = False

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        record = line[:6].strip().upper()

        if record == "MODEL":
            if not found_model:
                found_model = True
                in_model = True
                model_header = line
            else:
                break
            continue

        if record == "ENDMDL" and found_model and in_model:
            model_end = line
            in_model = False
            break

        if found_model and not in_model:
            continue

        selected.append(line)

    return selected, model_header, model_end


def extract_ligands(
    pdb_text: str,
    include_waters: bool = False,
    include_ions: bool = False,
) -> dict[str, str]:
    """Extract ligands into separate PDB strings indexed by ligand_id."""

    lines = [line.rstrip("\n") for line in pdb_text.splitlines()]
    model_lines, model_header, model_end = _iter_model_lines(lines)
    ligand_lines: dict[str, list[str]] = {}

    for line in model_lines:
        record = line[:6].strip().upper()
        if record not in {"HETATM", "ATOM"}:
            continue

        resname = line[17:20].strip()
        resname_upper = resname.upper()
        if record == "ATOM" and resname_upper in PROTEIN_RESNAMES:
            continue
        if not _should_include_resname(resname, include_waters, include_ions):
            continue

        ligand_id = _parse_ligand_id(line)
        ligand_lines.setdefault(ligand_id, []).append(line)

    extracted: dict[str, str] = {}
    for ligand_id, lines_for_ligand in ligand_lines.items():
        output_lines: list[str] = []
        if model_header:
            output_lines.append(model_header)
        output_lines.extend(lines_for_ligand)
        if model_end:
            output_lines.append(model_end)
        output_lines.append("END")
        extracted[ligand_id] = "\n".join(output_lines) + "\n"

    return extracted
