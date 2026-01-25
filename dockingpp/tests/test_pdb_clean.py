from dockingpp.data.pdb_clean import clean_pdb_text


PDB_SAMPLE = "\n".join(
    [
        "HEADER    TEST PDB",
        "ATOM      1  N   ALA A   1      11.104  13.207  10.456  1.00 20.00           N",
        "HETATM    2  O   HOH A   2      14.000  12.000   9.000  1.00 20.00           O",
        "HETATM    3  C   LIG A   3      10.000  10.000  10.000  1.00 20.00           C",
        "HETATM    4 ZN   ZN  A   4      12.000  10.000  11.000  1.00 20.00          ZN",
        "END",
    ]
)


def test_clean_pdb_removes_hetatm_waters_and_ions():
    cleaned = clean_pdb_text(PDB_SAMPLE)

    assert "ATOM" in cleaned
    assert "HOH" not in cleaned
    assert "LIG" not in cleaned
    assert "ZN" not in cleaned


def test_clean_pdb_keeps_allowed_hetatm_and_ions():
    cleaned = clean_pdb_text(
        PDB_SAMPLE,
        remove_waters=True,
        remove_hetatm=True,
        remove_ions=False,
        keep_het_resnames={"ZN"},
    )

    assert "ATOM" in cleaned
    assert "HOH" not in cleaned
    assert "LIG" not in cleaned
    assert "ZN" in cleaned
