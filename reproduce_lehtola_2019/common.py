import numpy as np
import pandas as pd

from psi4.core import Molecule
from lehtola_2019.molecules import load_molecules
from scf_guess.molecule import singlet, non_singlet, non_charged
from scf_guess.wavefunction import calculate_wavefunction, guess_wavefunction
from scf_guess.auxilary import file_cache
from scf_guess.metrics import f_score
from collections import defaultdict


@file_cache()
def score_molecule(molecule: Molecule, theory: str, guess: str, basis: str) -> float:
    guess = guess_wavefunction(molecule, guess, basis)
    reference, _, _ = calculate_wavefunction(molecule, theory, "SAP", basis)
    return f_score(guess, reference)


def clean_table(table: pd.DataFrame):
    table['SAP'] = table[["LDA-X", "CAP-X", "CHA-X"]].mean(axis=1)

    table.drop("GSZ", axis=1, inplace=True)
    table.drop("LDA-X", axis=1, inplace=True)
    table.drop("CAP-X", axis=1, inplace=True)
    table.drop("CHA-X", axis=1, inplace=True)


def reproduce_table(table: pd.DataFrame, theory_level: str, basis_set: str) -> pd.DataFrame:
    molecules = {m.name(): m for m in load_molecules() if non_charged(m)}

    def score(molecule: str, guess: str):
        return score_molecule(molecules[molecule], theory_level, guess, basis_set)

    return pd.DataFrame(
        [[score(row_label, col_label) for col_label in table.columns] for row_label in table.index],
        columns=table.columns,
        index=table.index
    )


def build_table_1(singlet: pd.DataFrame, non_singlet: pd.DataFrame) -> pd.DataFrame:
    table_1 = defaultdict(lambda: defaultdict(float))

    for variant, variant_name in zip([singlet, non_singlet], ["singlet", "non_singlet"]):
        for guess in variant.columns:
            table_1[guess][f"{variant_name}_min"] = np.min(variant[guess].values)
            table_1[guess][f"{variant_name}_mean"] = np.mean(variant[guess].values)

    return pd.DataFrame.from_dict(table_1, orient="index")
