import psi4
import numpy as np

from psi4.core import Matrix, Wavefunction

# TODO
# mints = psi4.core.MintsHelper(ref.basisset())
# S = mints.ao_overlap().to_array()

def f_score(guess: Wavefunction, reference: Wavefunction):
    Da_guess, Db_guess = guess.Da_subset("AO").np, guess.Db_subset("AO").np
    Da_ref, Db_ref = reference.Db_subset("AO").np, reference.Db_subset("AO").np

    S = Matrix(*Da_ref.shape)
    S.remove_symmetry(reference.S(), reference.aotoso().transpose())

    Q = lambda P_guess, P_ref: np.trace(P_guess @ S @ P_ref @ S)
    N = lambda P_ref: np.trace(P_ref @ S)

    numerator = Q(Da_guess, Da_ref) + Q(Db_guess, Db_ref)
    denominator = N(Da_ref) + N(Db_ref)

    return numerator / denominator


def diis_error(guess: Wavefunction, reference: Wavefunction):
    Da_guess, Db_guess = guess.Da_subset("AO").np, guess.Db_subset("AO").np
    Fa_guess, Fb_guess = guess.Fa_subset("AO").np, guess.Fb_subset("AO").np
    Da_ref, Db_ref = reference.Da_subset("AO").np, reference.Db_subset("AO").np

    S = Matrix(*Da_ref.shape)
    S.remove_symmetry(reference.S(), reference.aotoso().transpose())

    Ea = Fa_guess @ Da_guess @ S - S @ Da_guess @ Fa_guess
    Eb = Fb_guess @ Db_guess @ S - S @ Db_guess @ Fb_guess

    return np.trace(Ea @ Ea) + np.trace(Eb @ Eb)


def energy_error(guess: Wavefunction, reference: Wavefunction):
    with psi4.driver.p4util.hold_options_state():
        try:
            psi4.core.clean_options()
            psi4.core.clean()
            psi4.core.be_quiet()

            psi4.set_options({
                'basis': 'pcseg-0',
                'scf_type': 'pk',
                'maxiter': 0,
                'fail_on_maxiter': False,
                "reference": "rhf" if guess.molecule().multiplicity() == 1 else "uhf",
            })

            E_guess = psi4.energy('hf', molecule=guess.molecule())
            E_ref = reference.energy()
        finally:
            psi4.core.clean_options()
            psi4.core.clean()
            psi4.core.reopen_outfile()

    return E_guess / E_ref - 1.0
