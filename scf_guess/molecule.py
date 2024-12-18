import numpy as np

from psi4.core import Molecule
from typing import Callable
from numpy.typing import NDArray
from scf_guess.wavefunction import calculate_wavefunction


def charged(molecule: Molecule) -> bool:
    return molecule.molecular_charge() != 0


def non_charged(molecule: Molecule) -> bool:
    return not charged(molecule)


def singlet(molecule: Molecule) -> bool:
    return molecule.multiplicity() == 1


def non_singlet(molecule: Molecule) -> bool:
    return not singlet(molecule)


def degenerate(molecule: Molecule) -> bool:
    return _score_orbital_degeneracy(molecule) > 0


def non_degenerate(molecule: Molecule) -> bool:
    return not degenerate(molecule)


def categorize(molecules: list[Molecule], filters: list[Callable[[Molecule], bool]]) -> list[list[Molecule]]:
    categories = [list() for _ in range(len(filters))]

    for molecule in molecules:
        match = False

        for i, predicate in enumerate(filters):
            if not predicate(molecule): continue
            if match: raise RuntimeError("molecule matching multiple filters")

            categories[i].append(molecule)
            match = True

        if not match: raise RuntimeError("molecule matching no filter")

    return categories


def _find_close_energies(sequence: NDArray[float], delta: float = 1e-4) -> int:
    differences = np.abs(sequence[:, None] - sequence)
    upper_off_diagonal = np.triu(np.ones_like(differences, dtype=bool), k=1)
    js, ks = np.where((differences <= delta) & upper_off_diagonal)
    return len(js)


def _score_orbital_degeneracy(molecule: Molecule) -> int:
    wfn, _, _ = calculate_wavefunction(molecule, theory_level="HF", guess="SAP", basis_set="pcseg-0")
    score = 0

    for block, energies in enumerate(wfn.epsilon_a().to_array()):
        if not isinstance(energies, np.ndarray): continue
        score += _find_close_energies(energies)

    if molecule.multiplicity() != 1:
        for block, energies in enumerate(wfn.epsilon_b().to_array()):
            if not isinstance(energies, np.ndarray): continue
            score += _find_close_energies(energies)

    return score
