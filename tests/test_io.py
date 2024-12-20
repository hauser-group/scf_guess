from scf_guess.io import load_molecule


def test_load_molecule(tmp_path):
    xyz_string = """5
molecule "CrO4_2-" dataset "van Lenthe 2006" charge -2 multiplicity 1
Cr         0.00000        0.00000        0.00000
O          0.00000        0.00000        1.65206
O          1.55758        0.00000       -0.55068
O         -0.77879       -1.34891       -0.55068
O         -0.77879        1.34891       -0.55068
"""
    with open(tmp_path / "test.xyz", "w") as fout:
        fout.write(xyz_string)

    mol = load_molecule(tmp_path / "test.xyz")
    assert mol.molecular_charge() == -2
    assert mol.multiplicity() == 1
    assert mol.irrep_labels() == ["Ap", "App"]
    assert mol.point_group().symbol() == "cs"

    mol = load_molecule(tmp_path / "test.xyz", disable_symmetry=True)
    assert mol.irrep_labels() == ["A"]
    assert mol.point_group().symbol() == "c1"
