import pandas as pd
import numpy as np
import importlib.resources

from collections import defaultdict

tables = {
    "HF": {
        "sto-3g": {
            "singlet": 1,
            "non-singlet": 2
        },
        "pcseg-0": {
            "singlet": 5,
            "non-singlet": 6
        },
        "pcseg-1": {
            "singlet": 9,
            "non-singlet": 10
        },
        "aug-pcseg-2": {
            "singlet": 13,
            "non-singlet": 14
        }
    },
    "revTPSSh": {
        "sto-3g": {
            "singlet": 3,
            "non-singlet": 4
        },
        "pcseg-0": {
            "singlet": 7,
            "non-singlet": 8
        },
        "pcseg-1": {
            "singlet": 11,
            "non-singlet": 12
        },
        "aug-pcseg-2": {
            "singlet": 15,
            "non-singlet": 16
        }
    }
}


def load_table(theory: str, basis: str, variant: str) -> pd.DataFrame:
    base_path = importlib.resources.files(__package__) / "scores" / "tables"
    identifier = tables[theory][basis][variant]

    table = pd.read_csv(f"{base_path}/{identifier}.txt", skiprows=1, sep=r"\s+")

    table.set_index("Molecule", inplace=True)
    table.drop("Best", inplace=True)

    return table


def calculate_statistics(theory: str, basis: str) -> pd.DataFrame:
    statistics = defaultdict(lambda: defaultdict(float))

    for variant in tables[theory][basis].keys():
        table = load_table(theory, basis, variant)

        for guess in table.columns:
            statistics[guess][f"{variant}_min"] = np.min(table[guess].values)
            statistics[guess][f"{variant}_mean"] = np.mean(table[guess].values)

    return pd.DataFrame.from_dict(statistics, orient='index')
