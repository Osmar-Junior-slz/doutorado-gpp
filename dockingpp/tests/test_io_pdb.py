import numpy as np

from dockingpp.data.io import load_pdb_coords


def test_load_pdb_coords_reads_atom_and_hetatm(tmp_path):
    pdb_content = "\n".join(
        [
            "HEADER    TEST PDB",
            "ATOM      1  N   ALA A   1      11.104  13.207  10.456  1.00 20.00           N",
            "ATOM      2  CA  ALA A   1      12.560  13.500  10.300  1.00 20.00           C",
            "HETATM    3  O   HOH A   2      14.000  12.000   9.000  1.00 20.00           O",
            "TER",
            "END",
        ]
    )
    pdb_path = tmp_path / "sample.pdb"
    pdb_path.write_text(pdb_content, encoding="utf-8")

    coords = load_pdb_coords(str(pdb_path))

    assert coords.shape == (3, 3)
    assert np.allclose(
        coords,
        np.array(
            [
                [11.104, 13.207, 10.456],
                [12.56, 13.5, 10.3],
                [14.0, 12.0, 9.0],
            ]
        ),
    )
