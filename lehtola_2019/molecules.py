import os
import re
import importlib.resources

from psi4.core import Molecule
from glob import iglob


def load_molecule(path: str, disable_symmetry: bool = False) -> Molecule:
    with open(path, "r") as file:
        lines = file.readlines()

    q = re.search(r"charge\s+(-?\d+)", lines[1]).group(1)
    s = re.search(r"multiplicity\s+(\d+)", lines[1]).group(1)

    lines[1] = f"{q} {s}\n"
    xyz = "".join(lines)

    base_name = os.path.basename(path)
    file_name, _ = os.path.splitext(base_name)

    molecule = Molecule.from_string(xyz, name=file_name, dtype="xyz+")

    if disable_symmetry:
        molecule.reset_point_group("C1")

    return molecule


def load_molecules(disable_symmetry: bool = False) -> list[Molecule]:
    base_path = importlib.resources.files(__package__) / "geometries"
    paths = iglob(f"{base_path}/**/*.xyz", recursive=True)

    return [load_molecule(path, disable_symmetry=disable_symmetry) for path in paths]
