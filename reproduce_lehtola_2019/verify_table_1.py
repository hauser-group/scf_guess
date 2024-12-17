import pandas as pd

from lehtola_2019.scores import load_table
from common import build_table_1

if __name__ == "__main__":
    pd.options.display.float_format = '{:.3f}'.format

    theory_level = "HF"

    for basis_set in ["pcseg-0", "aug-pcseg-2"]:
        reference_table_singlet = load_table(theory_level, basis_set, "singlet")
        reference_table_non_singlet = load_table(theory_level, basis_set, "non_singlet")

        table_1 = build_table_1(reference_table_singlet, reference_table_non_singlet)
        print(f"Table 1 {basis_set} built from supporting infos:\n{table_1}")
