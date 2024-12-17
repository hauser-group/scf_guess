import psi4

from psi4.core import Molecule
from psi4.core import Wavefunction
from scf_guess.auxilary import clean_context, file_cache


@file_cache()
def calculate_wavefunction(molecule: Molecule, theory_level: str, guess: str, basis_set: str) -> Wavefunction:
    assert theory_level.lower() == "hf"
    print(f"calculating wavefunction for {molecule.name()}, guess={guess}, basis_set={basis_set}", end="", flush=True)

    with clean_context():
        stability_analysis = "NONE"
        if molecule.multiplicity() != 1:
            stability_analysis = "FOLLOW"
        elif molecule.natom() <= 30:
            stability_analysis = "CHECK"

        psi4.set_options({
            "BASIS": basis_set,
            "REFERENCE": "RHF" if molecule.multiplicity() == 1 else "UHF",
            "GUESS": guess,
            # Disable density fitting for highest possible accuracy and
            # because stability analysis is not available for density fitted
            # RHF wave functions:
            "SCF_TYPE": "PK",
            "STABILITY_ANALYSIS": stability_analysis
        })

        try:
            _, wfn = psi4.energy("hf", molecule=molecule, return_wfn=True)
        except psi4.ConvergenceError:
            # Try converging with second order SCF method
            psi4.set_options({
                "SOSCF": True,
                "SOSCF_START_CONVERGENCE": 1.0e-2,
                "SOSCF_MAX_ITER": 40,
            })

            _, wfn = psi4.energy("hf", molecule=molecule, return_wfn=True)

        print(": done")
        return wfn


@file_cache()
def guess_wavefunction(molecule: Molecule, guess: str, basis_set: str) -> Wavefunction:
    print(f"guessing wavefunction for {molecule.name()}, guess={guess}, basis_set={basis_set}", end="", flush=True)

    with clean_context():
        psi4.set_options({
            "BASIS": basis_set,
            "GUESS": guess
        })

        basis = psi4.core.BasisSet.build(molecule, target=basis_set)
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
