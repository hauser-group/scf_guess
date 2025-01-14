import psi4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from psi4.core import Molecule
from numpy.typing import NDArray
from lehtola_2019.molecules import load
from lehtola_2019.scores import load_table
from common import reproduce_table, clean_table
from collections import defaultdict
from scf_guess.wavefunction import calculate_wavefunction


def score_energy_degeneracy(sequence: NDArray[float], delta: float = 1e-4) -> int:
    differences = np.abs(sequence[:, None] - sequence)
    upper_off_diagonal = np.triu(np.ones_like(differences, dtype=bool), k=1)
    js, ks = np.where((differences <= delta) & upper_off_diagonal)
    return len(js)


def score_molecule_degeneracy(molecule: Molecule) -> int:
    wfn = calculate_wavefunction(molecule, theory_level="HF", guess="SAP", basis_set="pcseg-0")
    score = 0

    for block, energies in enumerate(wfn.epsilon_a().to_array()):
        if not isinstance(energies, np.ndarray): continue
        score += score_energy_degeneracy(energies)

    if molecule.multiplicity() != 1:
        for block, energies in enumerate(wfn.epsilon_b().to_array()):
            if not isinstance(energies, np.ndarray): continue
            score += score_energy_degeneracy(energies)

    return score


def categorize(relative_error: pd.DataFrame, molecules: dict):
    colors = {"non-degenerate": "blue", "degenerate": "red"}
    categories = defaultdict(list)

    for name, error in relative_error.items():
        molecule = molecules[name]
        category = "degenerate" if score_molecule_degeneracy(molecule) > 0 else "non-degenerate"
        categories[category].append(name)

    return [
        (categories[category], [relative_error[name] for name in categories[category]], color, category)
        for category, color in colors.items()
    ]

if __name__ == "__main__":
    pd.options.display.float_format = '{:.3f}'.format

    theory_level = "HF"
    basis_set = "pcseg-0"

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.be_quiet()

    reference_tables = [load_table(theory_level, basis_set, variant) for variant in ["singlet", "non_singlet"]]
    reference_table = pd.concat(reference_tables, axis=0)
    clean_table(reference_table)

    reproduced_table = reproduce_table(reference_table, theory_level, basis_set)
    relative_error = reproduced_table / reference_table - 1

    singlets, non_singlets = load()
    molecules = {molecule.name(): molecule for molecule in singlets + non_singlets}

    assert len(singlets) == 222
    assert len(non_singlets) == 37

    for name, molecule in molecules.items():
        degeneracy_score = score_molecule_degeneracy(molecule)
        if degeneracy_score > 0:
            print(f"{name} has degeneracy score {degeneracy_score}")

    guesses = relative_error.columns.values
    labels_ticks = {name: tick for tick, name in enumerate(relative_error.index.values)}

    fig, axs = plt.subplots(nrows=len(guesses), ncols=1, figsize=(8, 2 * len(guesses)), sharex=True)
    plt.xticks(ticks=list(labels_ticks.values()), labels=list(labels_ticks.keys()), rotation=90)

    for i, (guess, ax) in enumerate(zip(guesses, axs)):
        ax.set_title(f"{guess}")
        ax.set_ylabel("rel. err. / 1")

        for xs, ys, color, label in categorize(relative_error[guess], molecules):
            ax.scatter([labels_ticks[x] for x in xs], ys, color=color, label=label)

        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("identify_deviations.svg")
