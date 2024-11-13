import pytest
import psi4
import numpy as np
from scf_guess.metrics import f_score
from scipy.linalg import eigh


@pytest.mark.parametrize(
    "wfn_name, ref",
    [
        ("acetaldehyde_HF_STO-3G", 0.724147),
        ("oxetane_HF_STO-3G", 0.714633),
        ("methanol_HF_STO-3G", 0.766311),
    ],
)
def test_f_score(resource_path_root, wfn_name, ref, atol=1e-6):
    # Test 3 lines from Table S1 of Lehtola2019 (column CORE guess)
    wfn = psi4.core.Wavefunction.from_file(
        resource_path_root / "wavefunctions" / f"{wfn_name}.npy"
    )

    # Construct core Hamiltonian guess:
    S = wfn.S().np
    H = wfn.H().np
    _, vecs = eigh(H, S)

    Da_guess = vecs[:, : wfn.nalpha()] @ vecs[:, : wfn.nalpha()].T
    Db_guess = vecs[:, : wfn.nbeta()] @ vecs[:, : wfn.nbeta()].T

    np.testing.assert_allclose(f_score(S, wfn.Da().np, Da_guess), ref, atol=atol)
    np.testing.assert_allclose(
        f_score(S, wfn.Da().np, Da_guess, wfn.Db().np, Db_guess), ref, atol=atol
    )
