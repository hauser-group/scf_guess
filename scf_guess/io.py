import psi4


def load_molecule(path, disable_symmetry=False) -> psi4.core.Molecule:
    with open(path, "r") as fin:
        lines = fin.readlines()
    # read charge and multiplicity from the comment line and
    # replace the comment line with just these two integers
    split = lines[1].split()
    q = int(split[split.index("charge") + 1])
    s = int(split[split.index("multiplicity") + 1])
    lines[1] = f"{q} {s}\n"
    xyz_string = "".join(lines)

    mol = psi4.core.Molecule.from_string(xyz_string, dtype="xyz+")

    if disable_symmetry:
        mol.reset_point_group("C1")
    return mol
