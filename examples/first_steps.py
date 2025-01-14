# Investigate the hoclo molecule
# Plot the overlap and density matrices to first_steps.svg
# Calculate f-scores for all available guessing schemes

import psi4
import pandas as pd
import matplotlib.pyplot as plt

from lehtola_2019.molecules import load_molecules
from scf_guess.wavefunction import guess_wavefunction, calculate_wavefunction
from scf_guess.metrics import f_score

pd.options.display.float_format = '{:.3f}'.format

psi4.core.be_quiet()
psi4.set_memory("4 GiB")
psi4.set_num_threads(4)

theory = "HF"
guess = "SAD"
basis = "pcseg-0"

molecule = [m for m in load_molecules() if m.name() == "hoclo"][0]

guess_wfn = guess_wavefunction(molecule, guess=guess, basis=basis)
final_wfn, *_ = calculate_wavefunction(molecule, theory=theory, guess=guess, basis=basis)

fig, ax = plt.subplots(figsize=(12, 4), ncols=4, width_ratios=[1, 1, 1, 0.05])
imshow_kwargs = dict(vmin=-1.1, vmax=1.1, cmap="bwr")

ax[0].set_title("Overlap matrix")
ax[0].imshow(guess_wfn.S(), **imshow_kwargs)

ax[1].set_title(f"Guess density (f={f_score(guess_wfn, final_wfn):.4f})")
ax[1].imshow(guess_wfn.Da(), **imshow_kwargs)

ax[2].set_title("Converged density")
p = ax[2].imshow(final_wfn.Da(), **imshow_kwargs)

plt.colorbar(p, cax=ax[-1])
plt.savefig("first_steps.svg")

lehtola = pd.DataFrame({
    "GWH": [0.628],
    "CORE": [0.740],
    "SAD": [0.936],
    "SADNO": [0.977],
    "HUCKEL": [0.984],
    "GSZ": [0.974],
    "LDA-X": [0.994],
    "CAP-X": [0.993],
    "CHA-X": [0.994]
}, index=["hoclo"])

print(f"f-scores from Lehtola:\n{lehtola}")

ours = {}

for guess in ["GWH", "CORE", "SAD", "SADNO", "HUCKEL", "SAP", "SAPGAU"]:
    guess_wfn = guess_wavefunction(molecule, guess=guess, basis=basis)
    ours[guess] = [f_score(guess_wfn, final_wfn)]

print(f"\nf-scores computed by us:\n{pd.DataFrame(ours, index=['hoclo'])}")
