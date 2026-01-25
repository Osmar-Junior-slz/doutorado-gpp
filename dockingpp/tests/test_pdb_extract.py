from dockingpp.data.pdb_extract import extract_ligands


PDB_SAMPLE = "\n".join(
    [
        "HEADER    TEST PDB",
        "ATOM      1  N   ALA A   1      11.104  13.207  10.456  1.00 20.00           N",
        "HETATM    2  O   HOH A   2      14.000  12.000   9.000  1.00 20.00           O",
        "HETATM    3  C   LIG A 401      10.000  10.000  10.000  1.00 20.00           C",
        "HETATM    4  C   LIG B 402      10.100  10.100  10.100  1.00 20.00           C",
        "HETATM    5  C   DRG A 500      11.000  10.000  10.000  1.00 20.00           C",
        "HETATM    6 ZN   ZN  A 900      12.000  10.000  11.000  1.00 20.00          ZN",
        "END",
    ]
)


def test_extract_ligands_default_filters():
    ligands = extract_ligands(PDB_SAMPLE, include_waters=False, include_ions=False)

    assert set(ligands.keys()) == {"LIG_A_401", "LIG_B_402", "DRG_A_500"}

    lig_a = ligands["LIG_A_401"]
    assert "LIG A 401" in lig_a
    assert "LIG B 402" not in lig_a
    assert "DRG A 500" not in lig_a

    lig_b = ligands["LIG_B_402"]
    assert "LIG B 402" in lig_b
    assert "LIG A 401" not in lig_b
    assert "DRG A 500" not in lig_b

    drg = ligands["DRG_A_500"]
    assert "DRG A 500" in drg
    assert "LIG A 401" not in drg
    assert "LIG B 402" not in drg


def test_extract_ligands_includes_ions_when_requested():
    ligands = extract_ligands(PDB_SAMPLE, include_ions=True)

    assert "ZN_A_900" in ligands
    assert "ZN  A 900" in ligands["ZN_A_900"]
