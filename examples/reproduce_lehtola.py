#!/usr/bin/env python3
from itertools import product

import pandas as pd
import psi4
from numpy.typing import NDArray
from psi4.core import Molecule, Matrix, Wavefunction
import numpy as np
from scf_guess.metrics import f_score
from filecache import filecache
from statistics import mean
from lehtola_2019.molecules import load_molecules
from lehtola_2019.scores import calculate_statistics, load_table
from scf_guess.wavefunction import calculate_wavefunction, guess_wavefunction
from scf_guess.auxilary import clean_context, Cacheable, file_cache
from collections import defaultdict





psi4.core.clean()
psi4.core.clean_options()

psi4.core.be_quiet()
psi4.set_memory("20 GB")
psi4.set_num_threads(8)
print(psi4.get_memory())

psi4.core.clean()
psi4.core.clean_options()
print(psi4.get_memory())

exit()

singlets, non_singlets = load()

assert len(singlets) == 222
assert len(non_singlets) == 37




def get_guess(molecule: Molecule, guess: str, basis_set: str):
    wavefunction = guess_wavefunction(molecule, guess=guess, basis=basis_set)
    return wavefunction.Da_subset("AO").np, wavefunction.Db_subset("AO").np


def get_reference(molecule: Molecule, guess: str, basis_set: str):
    wavefunction = calculate_wavefunction(molecule, guess=guess, basis_set=basis_set)
    Da, Db = wavefunction.Da_subset("AO").np, wavefunction.Db_subset("AO").np
    S = psi4.core.Matrix(*Da.shape)
    S.remove_symmetry(wavefunction.S(), wavefunction.aotoso().transpose())
    return Da, Db, S





def score_molecule(mol, guess, basis_set):
    Da_guess, Db_guess = get_guess(mol, guess=guess, basis_set=basis_set)
    Da_scf, Db_scf, S = get_reference(mol, guess="SAP", basis_set=basis_set)
    return f_score(psi4.core.Matrix.from_array(S), Da_scf, Da_guess, Db_scf, Db_guess)


def score_data_set(molecules, guess, basis_set):
    f_scores = np.empty(len(molecules))
    f_scores.fill(np.nan)

    for i, molecule in enumerate(molecules):
        f_scores[i] = score_molecule(molecule, guess, basis_set)

    return f_scores

pd.options.display.float_format = '{:.3f}'.format

basis_set = "pcseg-0"

reference_statistics = calculate_statistics("HF", basis_set)
# print(reference_statistics)
# exit()

merged_row = {
    "singlet_min": reference_statistics.iloc[-3:]["singlet_min"].min(),
    "singlet_mean": reference_statistics.iloc[-3:]["singlet_mean"].mean(),
    "non_singlet_min": reference_statistics.iloc[-3:]["non_singlet_min"].min(),
    "non_singlet_mean": reference_statistics.iloc[-3:]["non_singlet_mean"].mean(),
}

reference_statistics.drop("GSZ", inplace=True)
reference_statistics.drop("LDA-X", inplace=True)
reference_statistics.drop("CAP-X", inplace=True)
reference_statistics.drop("CHA-X", inplace=True)

reference_statistics.loc["SAP"] = merged_row
print(f"reference statistics for {basis_set}:\n{reference_statistics}")


def calculate_f_score(guess, type, value, ref_stat, basis_set=basis_set):
    assert type in ("singlet", "non_singlet")

    f_scores = score_data_set(singlets if type == "singlet" else non_singlets, guess=guess, basis_set=basis_set)
    f_score = np.nanmin(f_scores) if value == "min" else np.nanmean(f_scores)

    reference_f_score = ref_stat.loc[guess, f"{type}_{value}"]
    f_score_error = abs((reference_f_score - f_score) / reference_f_score)

    return f_score, f_score_error*100


for guess in reference_statistics.index:
    f_singlet_min, f_singlet_min_err = calculate_f_score(guess, "singlet", "min", reference_statistics)
    f_singlet_mean, f_singlet_mean_err = calculate_f_score(guess, "singlet", "mean", reference_statistics)

    f_non_singlet_min, f_non_singlet_min_err = calculate_f_score(guess, "non_singlet", "min", reference_statistics)
    f_non_singlet_mean, f_non_singlet_mean_err = calculate_f_score(guess, "non_singlet", "mean", reference_statistics)

    print(
        f"{guess:>8s}:",
        f"{f_singlet_min:.3f}({f_singlet_min_err:3.0f}%)  {f_singlet_mean:.3f}({f_singlet_mean_err:3.0f}%)  ",
        f"{f_non_singlet_min:.3f}({f_non_singlet_min_err:3.0f}%)  {f_non_singlet_mean:.3f}({f_non_singlet_mean_err:3.0f}%)",
    )
exit()


def score_energy_degeneracy(sequence: NDArray[float], delta: float = 1e-5) -> int:
    differences = np.abs(sequence[:, None] - sequence)
    upper_off_diagonal = np.triu(np.ones_like(differences, dtype=bool), k=1)
    js, ks = np.where((differences <= delta) & upper_off_diagonal)
    return len(js)


def score_molecule_degeneracy(molecule: Molecule) -> int:
    ref_wfn = calculate_wavefunction(molecule, guess="SAP", basis_set="pcseg-0")
    degen = 0
    for block, energies in enumerate(ref_wfn.epsilon_a().to_array()):
        if not isinstance(energies, np.ndarray): continue
        degen += score_energy_degeneracy(energies)
    if molecule.multiplicity() != 1:
        for block, energies in enumerate(ref_wfn.epsilon_b().to_array()):
            if not isinstance(energies, np.ndarray): continue
            degen += score_energy_degeneracy(energies)
    return degen


hf_pcseg0_singlets = load_table("HF", "pcseg-0", "singlet")
hf_pcseg0_singlets.drop("LDA-X", axis=1, inplace=True)
hf_pcseg0_singlets.drop("CAP-X", axis=1, inplace=True)
hf_pcseg0_singlets.drop("CHA-X", axis=1, inplace=True)
hf_pcseg0_singlets.drop("GSZ", axis=1, inplace=True)

singlets_nondegen = []
for s in singlets:
    if score_molecule_degeneracy(s) == 0:
        singlets_nondegen.append(s)
    else:
        hf_pcseg0_singlets.drop(s.name(), axis=0, inplace=True)

hf_pcseg0_non_singlets = load_table("HF", "pcseg-0", "non_singlet")
hf_pcseg0_non_singlets.drop("LDA-X", axis=1, inplace=True)
hf_pcseg0_non_singlets.drop("CAP-X", axis=1, inplace=True)
hf_pcseg0_non_singlets.drop("CHA-X", axis=1, inplace=True)
hf_pcseg0_non_singlets.drop("GSZ", axis=1, inplace=True)

non_singlets_nondegen = []
for s in non_singlets:
    if score_molecule_degeneracy(s) == 0:
        non_singlets_nondegen.append(s)
    else:
        hf_pcseg0_non_singlets.drop(s.name(), axis=0, inplace=True)

print("got non-degenerate molecules")
print(len(singlets_nondegen))
print(len(hf_pcseg0_singlets))
print(len(non_singlets_nondegen))
print(len(hf_pcseg0_non_singlets))

def mod_calculate_statistics(singlets, non_singlets) -> pd.DataFrame:
    statistics = defaultdict(lambda: defaultdict(float))

    for guess in singlets.columns:
        statistics[guess][f"singlet_min"] = np.min(singlets[guess].values)
        statistics[guess][f"singlet_mean"] = np.mean(singlets[guess].values)

    for guess in non_singlets.columns:
        statistics[guess][f"non_singlet_min"] = np.min(non_singlets[guess].values)
        statistics[guess][f"non_singlet_mean"] = np.mean(non_singlets[guess].values)

    return pd.DataFrame.from_dict(statistics, orient='index')

print("the modified statistics without degenerate molecules")
hf_pcseg0_statistics = mod_calculate_statistics(hf_pcseg0_singlets, hf_pcseg0_non_singlets)
print(hf_pcseg0_statistics)
#exit()


# for guess in hf_pcseg0_statistics.index:
#     f_singlet_min, f_singlet_min_err = calculate_f_score(guess, "singlet", "min", hf_pcseg0_statistics)
#     f_singlet_mean, f_singlet_mean_err = calculate_f_score(guess, "singlet", "mean", hf_pcseg0_statistics)
#
#     f_non_singlet_min, f_non_singlet_min_err = calculate_f_score(guess, "non_singlet", "min", hf_pcseg0_statistics)
#     f_non_singlet_mean, f_non_singlet_mean_err = calculate_f_score(guess, "non_singlet", "mean", hf_pcseg0_statistics)
#
#     print(
#         f"{guess:>8s}:",
#         f"{f_singlet_min:.3f}({f_singlet_min_err:3.0f}%)  {f_singlet_mean:.3f}({f_singlet_mean_err:3.0f}%)  ",
#         f"{f_non_singlet_min:.3f}({f_non_singlet_min_err:3.0f}%)  {f_non_singlet_mean:.3f}({f_non_singlet_mean_err:3.0f}%)",
#     )


print(hf_pcseg0_singlets)
overall = defaultdict(list)

for mol in singlets_nondegen:
    for guess in hf_pcseg0_singlets.columns:
        lehtola_f_score = hf_pcseg0_singlets.loc[mol.name(), guess]
        my_f_score = score_molecule(mol, guess, basis_set="pcseg-0")

        rel_err = abs(lehtola_f_score - my_f_score) / lehtola_f_score * 100
        desc = f"{rel_err:.0f} {mol.name()} {guess} lehtola={lehtola_f_score} my={my_f_score}"

        if rel_err > 0:
            overall[guess].append(desc)

print("only singlet and non-degenerate")

for guess, descs in overall.items():
    print(f"GUESS {guess}:")
    for d in descs:
        print(d)

    print()

# def load_table(theory_level: str, basis_set: str, variant: str) -> pd.DataFrame:
#     base_path = importlib.resources.files(__package__) / "scores" / "tables"
#     identifier = tables[theory_level][basis_set][variant]
#
#     table = pd.read_csv(f"{base_path}/{identifier}.txt", skiprows=1, sep=r"\s+")
#
#     table.set_index("Molecule", inplace=True)
#     table.drop("Best", inplace=True)
#
#     return table

# def score_molecule(mol, guess, basis_set):
#     Da_guess, Db_guess = get_guess(mol, guess=guess, basis_set=basis_set)
#     Da_scf, Db_scf, S = get_reference(mol, guess="SAP", basis_set=basis_set)
#     return f_score(psi4.core.Matrix.from_array(S), Da_scf, Da_guess, Db_scf, Db_guess)
#
# def score_data_set(molecules, guess, basis_set):
#     f_scores = np.empty(len(molecules))
#     f_scores.fill(np.nan)
#
#     for i, molecule in enumerate(molecules):
#         f_scores[i] = score_molecule(molecule, guess, basis_set)
#
#     return f_scores
