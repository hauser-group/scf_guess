import pandas as pd
import numpy as np
import importlib.resources

from collections import defaultdict

tables = {
    "HF": {
        "STO-3G": {
            "singlet": 1,
            "non_singlet": 2
        },
        "pcseg-0": {
            "singlet": 5,
            "non_singlet": 6
        },
        "pcseg-1": {
            "singlet": 9,
            "non_singlet": 10
        },
        "aug-pcseg-2": {
            "singlet": 13,
            "non_singlet": 14
        }
    },
    "revTPSSh": {
        "STO-3G": {
            "singlet": 3,
            "non_singlet": 4
        },
        "pcseg-0": {
            "singlet": 7,
            "non_singlet": 8
        },
        "pcseg-1": {
            "singlet": 11,
            "non_singlet": 12
        },
        "aug-pcseg-2": {
            "singlet": 15,
            "non_singlet": 16
        }
    }
}


def load_table(theory_level: str, basis_set: str, variant: str) -> pd.DataFrame:
    base_path = importlib.resources.files(__package__) / "scores" / "tables"
    identifier = tables[theory_level][basis_set][variant]

    table = pd.read_csv(f"{base_path}/{identifier}.txt", skiprows=1, sep=r"\s+")

    table.set_index("Molecule", inplace=True)
    table.drop("Best", inplace=True)

    return table


def calculate_statistics(theory_level: str, basis_set: str) -> pd.DataFrame:
    statistics = defaultdict(lambda: defaultdict(float))

    for variant in tables[theory_level][basis_set].keys():
        table = load_table(theory_level, basis_set, variant)

        for guess in table.columns:
            statistics[guess][f"{variant}_min"] = np.min(table[guess].values)
            statistics[guess][f"{variant}_mean"] = np.mean(table[guess].values)

    return pd.DataFrame.from_dict(statistics, orient='index')
