import psi4
import pandas as pd

from lehtola_2019.scores import load_table
from common import build_table_1, reproduce_table, clean_table


if __name__ == "__main__":
    pd.options.display.float_format = '{:.3f}'.format

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.be_quiet()

    psi4.set_memory("20 GB")
    psi4.set_num_threads(8)

    theory_level = "HF"
    basis_set = "pcseg-0"

    reference_table_singlet = load_table(theory_level, basis_set, "singlet")
    clean_table(reference_table_singlet)
    reference_table_non_singlet = load_table(theory_level, basis_set, "non_singlet")
    clean_table(reference_table_non_singlet)
    reference_table_1 = build_table_1(reference_table_singlet, reference_table_non_singlet)

    reproduced_table_singlet = reproduce_table(reference_table_singlet, theory_level, basis_set)
    reproduced_table_non_singlet = reproduce_table(reference_table_non_singlet, theory_level, basis_set)
    reproduced_table_1 = build_table_1(reproduced_table_singlet, reproduced_table_non_singlet)

    print(f"Table 1 {basis_set} built from supporting infos:\n{reference_table_1}")
    print(f"Table 1 {basis_set} reproduced:\n{reproduced_table_1}")
    print(f"Relative error:\n{reproduced_table_1/reference_table_1 - 1}")
