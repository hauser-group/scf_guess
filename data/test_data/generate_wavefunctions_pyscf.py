import numpy as np
from glob import glob
from pathlib import Path
from pyscf import gto, scf


def evaluate_xyz(xyz_path, basisset="STO-3G"):
    with open(xyz_path, "r") as fin:
        lines = fin.readlines()
    # read charge and multiplicity from the comment line and
    # replace the comment line with just these two integers
    split = lines[1].split()
    q = int(split[split.index("charge") + 1])
    s = int(split[split.index("multiplicity") + 1]) - 1
    mol = gto.M(atom=xyz_path, basis=basisset, spin=s, charge=q)

    if s == 0:
        mf = scf.RHF(mol).run()
    else:
        mf = scf.UHF(mol).run()

    S = mf.get_ovlp()

    if mol.natm <= 30:
        # Run stability analysis for the SCF wave function
        mo1, _, stable, _ = mf.stability(return_status=True)
        while not stable:
            dm1 = mf.make_rdm1(mo1, mf.mo_occ)
            mf = mf.run(dm1)
            mo1, _, stable, _ = mf.stability(return_status=True)

    dm = mf.make_rdm1()
    if s == 0:
        dm = np.stack([dm, dm], axis=0) / 2
    return S, dm


def evaluate_geometries(basisset="STO-3G"):
    xyz_paths = glob("geometries/**/*.xyz")

    wfn_dir = Path(f"wavefunctions_pyscf/HF/{basisset}")

    for xyz_path in xyz_paths:
        s = xyz_path.split("/")
        dataset_name = s[1]
        folder_path = wfn_dir / dataset_name
        folder_path.mkdir(parents=True, exist_ok=True)
        name = s[-1].removesuffix(".xyz")
        wfn_path = folder_path / f"{name}.npz"
        if wfn_path.exists():
            continue
        print(f"{dataset_name}/{name}")

        S, dm = evaluate_xyz(xyz_path, basisset)
        np.savez(wfn_path, S=S, dm=dm)


if __name__ == "__main__":
    basisset = "pcseg-0"

    evaluate_geometries(basisset=basisset)

# if __name__ == "__main__":
#     basisset = "pcseg-0"

#     S, dm = evaluate_xyz("geometries/W4-17/ch.xyz", basisset=basisset)
#     print(S.shape, dm.shape)
