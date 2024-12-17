import numpy as np
import pandas as pd

from psi4.core import Molecule, Matrix
from scf_guess.metrics import f_score
from lehtola_2019.molecules import load
from scf_guess.wavefunction import calculate_wavefunction, guess_wavefunction
from scf_guess.auxilary import file_cache
from collections import defaultdict


def get_guess(molecule: Molecule, guess: str, basis_set: str):
    wavefunction = guess_wavefunction(molecule, guess=guess, basis_set=basis_set)
    return wavefunction.Da_subset("AO").np, wavefunction.Db_subset("AO").np


def get_reference(molecule: Molecule, theory_level: str, guess: str, basis_set: str):
    wavefunction = calculate_wavefunction(molecule, theory_level=theory_level,  guess=guess, basis_set=basis_set)
    Da, Db = wavefunction.Da_subset("AO").np, wavefunction.Db_subset("AO").np
    S = Matrix(*Da.shape)
    S.remove_symmetry(wavefunction.S(), wavefunction.aotoso().transpose())
    return Da, Db, S


@file_cache()
def score_molecule(mol, theory_level, guess, basis_set) -> float:
    Da_guess, Db_guess = get_guess(mol, guess=guess, basis_set=basis_set)
    Da_scf, Db_scf, S = get_reference(mol, theory_level=theory_level, guess="SAP", basis_set=basis_set)
    return f_score(Matrix.from_array(S), Da_scf, Da_guess, Db_scf, Db_guess)


def clean_table(table: pd.DataFrame):
    table['SAP'] = table[["LDA-X", "CAP-X", "CHA-X"]].mean(axis=1)

    table.drop("GSZ", axis=1, inplace=True)
    table.drop("LDA-X", axis=1, inplace=True)
    table.drop("CAP-X", axis=1, inplace=True)
    table.drop("CHA-X", axis=1, inplace=True)


def reproduce_table(table: pd.DataFrame, theory_level: str, basis_set: str) -> pd.DataFrame:
    singlets, non_singlets = load()
    molecules = {molecule.name(): molecule for molecule in singlets + non_singlets}

    assert len(singlets) == 222
    assert len(non_singlets) == 37

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
