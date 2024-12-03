import os
import re

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


def load(disable_symmetry: bool = False) -> tuple[list[Molecule], list[Molecule]]:
    paths = iglob("geometries/**/*.xyz", recursive=True)

    molecules = [load_molecule(path, disable_symmetry=disable_symmetry) for path in paths]
    singlets, non_singlets = [], []

    for molecule in molecules:
        if molecule.molecular_charge() != 0:
            continue

        category = singlets if molecule.multiplicity() == 1 else non_singlets
        category.append(molecule)

    return singlets, non_singlets
