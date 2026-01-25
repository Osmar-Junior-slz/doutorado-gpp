"""Utilities for cleaning PDB text."""

from __future__ import annotations

from typing import Iterable


WATER_RESNAMES = {"HOH", "WAT", "SOL", "H2O"}
ION_RESNAMES = {
    "NA",
    "CL",
    "K",
    "CA",
    "MG",
    "ZN",
    "MN",
    "FE",
    "CU",
    "CO",
    "NI",
    "CD",
    "HG",
    "BR",
    "I",
    "PO4",
    "SO4",
}
HEADER_RECORDS = {"HEADER", "TITLE", "REMARK", "MODEL", "ENDMDL", "TER", "END"}


def _get_record(line: str) -> str:
    return line[:6].strip().upper()


def _get_resname(line: str) -> str:
    return line[17:20].strip().upper()


def _should_keep_atom(
    record: str,
    resname: str,
    remove_waters: bool,
    remove_hetatm: bool,
    remove_ions: bool,
    keep_het_resnames: set[str] | None,
) -> bool:
    if remove_waters and resname in WATER_RESNAMES:
        return False
    if remove_ions and resname in ION_RESNAMES:
        return False
    if record == "HETATM" and remove_hetatm:
        if keep_het_resnames and resname in keep_het_resnames:
            return True
        return False
    return True


def _iter_cleaned_lines(
    lines: Iterable[str],
    remove_waters: bool,
    remove_hetatm: bool,
    remove_ions: bool,
    keep_het_resnames: set[str] | None,
) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        if not line:
            continue
        record = _get_record(line)
        if record in {"ATOM", "HETATM"}:
            resname = _get_resname(line)
            if _should_keep_atom(
                record=record,
                resname=resname,
                remove_waters=remove_waters,
                remove_hetatm=remove_hetatm,
                remove_ions=remove_ions,
                keep_het_resnames=keep_het_resnames,
            ):
                cleaned.append(line)
            continue
        if record in HEADER_RECORDS:
            cleaned.append(line)
    return cleaned


def clean_pdb_text(
    pdb_text: str,
    remove_waters: bool = True,
    remove_hetatm: bool = True,
    remove_ions: bool = True,
    keep_het_resnames: set[str] | None = None,
) -> str:
    """Return cleaned PDB content based on simple deterministic filtering."""

    keep_het_resnames = {name.strip().upper() for name in keep_het_resnames or set() if name.strip()}
    lines = [line.rstrip("\n") for line in pdb_text.splitlines()]
    cleaned = _iter_cleaned_lines(
        lines,
        remove_waters=remove_waters,
        remove_hetatm=remove_hetatm,
        remove_ions=remove_ions,
        keep_het_resnames=keep_het_resnames or None,
    )
    if not cleaned or cleaned[-1] != "END":
        cleaned.append("END")
    return "\n".join(cleaned) + "\n"
