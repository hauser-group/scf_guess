# Investigate the acetaldehyde molecule
# Plot the overlap and density matrices to second_steps.svg for the individual irreps
# Calculate f-scores for all available guessing schemes

import psi4
import pandas as pd
import matplotlib.pyplot as plt

from lehtola_2019.molecules import load_molecules
from scf_guess.wavefunction import guess_wavefunction, calculate_wavefunction
from scf_guess.metrics import f_score

psi4.core.be_quiet()
psi4.set_memory("4 GiB")
psi4.set_num_threads(4)

theory = "HF"
guess = "SAD"
basis = "pcseg-0"

molecule = [m for m in load_molecules() if m.name() == "acetaldehyde"][0]

guess_wfn = guess_wavefunction(molecule, guess=guess, basis=basis)
final_wfn, *_ = calculate_wavefunction(molecule, theory=theory, guess=guess, basis=basis)

irreps = guess_wfn.S().nirrep()

fig, ax = plt.subplots(figsize=(12, 4), ncols=4, nrows=irreps+1, width_ratios=[1, 1, 1, 0.05])
imshow_kwargs = dict(vmin=-1.1, vmax=1.1, cmap="bwr")

for irrep in range(irreps):
    ax[irrep][0].set_title(f"Overlap matrix (irrep {irrep})")
    ax[irrep][0].imshow(guess_wfn.S().to_array(dense=False)[irrep], **imshow_kwargs)

    ax[irrep][1].set_title(f"Guess density (irrep {irrep})")
    ax[irrep][1].imshow(guess_wfn.Da().to_array(dense=False)[irrep], **imshow_kwargs)

    ax[irrep][2].set_title(f"Converged density (irrep {irrep})")
    p = ax[irrep][2].imshow(final_wfn.Da().to_array(dense=False)[irrep], **imshow_kwargs)

    plt.colorbar(p, cax=ax[irrep][3])

ax[-1][0].set_title("Overlap matrix")
ax[-1][0].imshow(guess_wfn.S().to_array(dense=True), **imshow_kwargs)

ax[-1][1].set_title(f"Guess density (f={f_score(guess_wfn, final_wfn):.4f})")
ax[-1][1].imshow(guess_wfn.Da().to_array(dense=True), **imshow_kwargs)

p = ax[-1][2].imshow(final_wfn.Da().to_array(dense=True), **imshow_kwargs)
ax[-1][2].set_title("Converged density")

plt.colorbar(p, cax=ax[-1][3])
plt.savefig("second_steps.svg")

lehtola = pd.DataFrame({
    "GWH": [0.484],
    "CORE": [0.618],
    "SAD": [0.893],
    "SADNO": [0.968],
    "HUCKEL": [0.973],
    "GSZ": [0.913],
    "LDA-X": [0.977],
    "CAP-X": [0.976],
    "CHA-X": [0.979]
}, index=["acetaldehyde"])

print(f"f-scores from Lehtola:\n{lehtola}")

ours = {}

for guess in ["GWH", "CORE", "SAD", "SADNO", "HUCKEL", "SAP", "SAPGAU"]:
    guess_wfn = guess_wavefunction(molecule, guess=guess, basis=basis)
    ours[guess] = [f_score(guess_wfn, final_wfn)]

print(f"\nf-scores computed by us:\n{pd.DataFrame(ours, index=['acetaldehyde'])}")
