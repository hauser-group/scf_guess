import psi4
import re
import os

from typing import Tuple
from psi4.core import Molecule, Wavefunction
from scf_guess.auxilary import clean_context, file_cache


def _hartree_fock(molecule: Molecule, guess: str, basis: str, second_order: bool = False) -> Wavefunction:
    stability_analysis = "NONE"
    if molecule.multiplicity() != 1:
        stability_analysis = "FOLLOW"
    elif molecule.natom() <= 30:
        stability_analysis = "CHECK"

    psi4.set_options({
        "BASIS": basis,
        "REFERENCE": "RHF" if molecule.multiplicity() == 1 else "UHF",
        "GUESS": guess,
        # Disable density fitting for highest possible accuracy and
        # because stability analysis is not available for density fitted
        # RHF wave functions:
        "SCF_TYPE": "PK",
        "STABILITY_ANALYSIS": stability_analysis
    })

    if second_order:
        psi4.set_options({
            "SOSCF": True,
            "SOSCF_START_CONVERGENCE": 1.0e-2,
            "SOSCF_MAX_ITER": 40,
        })

    _, wfn = psi4.energy("HF", molecule=molecule, return_wfn=True)
    return wfn


def _hartree_fock_iterations(stdout_file: str) -> int:
    iterations = []

    with open(stdout_file, "r") as stdout:
        pattern = re.compile(r"^\S+\s+iter\s+(\d+):")
        count = False

        for line in stdout:
            if "Unable to find file" in line:
                raise RuntimeError(line)

            if "Failed to converge" in line:
                count = False
                iterations = []
                continue

            if line.strip() == "==> Iterations <==":
                count = True
                continue

            if count and line.strip().startswith("==>"):
                count = False
                continue

            if count:
                iteration = pattern.match(line.strip())
                if iteration is not None:
                    iterations.append(int(iteration.group(1)))

    # offset = None
    #
    # for i, iteration in enumerate(iterations):
    #     if offset is None:
    #         offset = i - iteration
    #
    #     if i != iteration - 1:
    #         raise RuntimeError(f"extracted invalid iteration sequence {iterations}")

    return len(iterations)


@file_cache()
def calculate_wavefunction(molecule: Molecule, theory: str, guess: str | Wavefunction, basis: str) \
        -> Tuple[Wavefunction, int, bool]:
    assert theory.upper() == "HF"
    print(f"calculating wavefunction for {molecule.name()}, guess={guess}, basis={basis}", end="", flush=True)

    if isinstance(guess, Wavefunction):
        # See https://forum.psicode.org/t/custom-guess-for-hartree-fock/2026/6
        scratch_dir = psi4.core.IOManager.shared_object().get_default_path()
        guess.to_file(filename=f"{scratch_dir}/stdout.{molecule.name()}.{os.getpid()}.180.npy")

    try:
        with clean_context() as stdout_file:
            second_order = False
            wfn = _hartree_fock(molecule, guess if isinstance(guess, str) else "READ", basis, second_order)
            iterations = _hartree_fock_iterations(stdout_file)
    except psi4.ConvergenceError:
        with clean_context() as stdout_file:
            second_order = True
            wfn = _hartree_fock(molecule, guess if isinstance(guess, str) else "READ", basis, second_order)
            iterations = _hartree_fock_iterations(stdout_file)

    print(": done")
    return wfn, iterations, second_order


@file_cache()
def guess_wavefunction(molecule: Molecule, guess: str, basis: str) -> Wavefunction:
    print(f"guessing wavefunction for {molecule.name()}, guess={guess}, basis={basis}", end="", flush=True)

    with clean_context():
        psi4.set_options({
            "BASIS": basis,
            "GUESS": guess
        })

        basis = psi4.core.BasisSet.build(molecule, target=basis)
        ref_wfn = psi4.core.Wavefunction.build(molecule, basis)
        start_wfn = psi4.driver.scf_wavefunction_factory(
            name="hf",
            ref_wfn=ref_wfn,
            reference="RHF" if molecule.multiplicity == 1 else "UHF",
        )
        start_wfn.form_H()
        start_wfn.form_Shalf()
        start_wfn.guess()

        print(": done")
        return start_wfn
