import psi4
from glob import glob
from scf_guess.io import load_molecule
from pathlib import Path


def evaluate_geometries(basisset="STO-3G"):
    xyz_paths = glob("geometries/**/*.xyz")

    wfn_dir = Path(f"wavefunctions/HF/{basisset}")

    for xyz_path in xyz_paths:
        s = xyz_path.split("/")
        dataset_name = s[1]
        folder_path = wfn_dir / dataset_name
        folder_path.mkdir(parents=True, exist_ok=True)
        name = s[-1].removesuffix(".xyz")
        wfn_path = folder_path / f"{name}.npy"
        if wfn_path.exists():
            continue
        print(f"{dataset_name}/{name}")

        mol = load_molecule(xyz_path)
        stability_analysis = "NONE"
        if mol.multiplicity() != 1:
            stability_analysis = "FOLLOW"
        elif mol.natom() <= 30:
            stability_analysis = "CHECK"
        psi4.set_options(
            {
                "BASIS": basisset,
                "REFERENCE": "RHF" if mol.multiplicity() == 1 else "UHF",
                "GUESS": "SAP",
                # Disable density fitting for highest possible accuracy and
                # because stability analysis is not available for density fitted
                # RHF wave functions:
                "SCF_TYPE": "PK",
                "STABILITY_ANALYSIS": stability_analysis,
            }
        )

        try:
            _, wfn = psi4.energy(name="hf", molecule=mol, return_wfn=True)
        except psi4.ConvergenceError:
            # Try converging with second order SCF method
            psi4.set_options(
                {
                    "SOSCF": True,
                    "SOSCF_START_CONVERGENCE": 1.0e-2,
                    "SOSCF_MAX_ITER": 40,
                }
            )
            _, wfn = psi4.energy(name="hf", molecule=mol, return_wfn=True)
        finally:
            # Reset for the next example
            psi4.core.clean_options()
            psi4.core.clean()
        wfn.to_file(str(wfn_path))


if __name__ == "__main__":
    basisset = "pcseg-0"

    psi4.set_memory("24 GiB")
    psi4.set_num_threads(8)
    psi4.set_output_file(f"{basisset}_wfn.log", append=True)

    evaluate_geometries(basisset=basisset)
