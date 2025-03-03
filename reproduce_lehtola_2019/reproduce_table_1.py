import pandas as pd

from itertools import product
from lehtola_2019.scores import load_table
from lehtola_2019.molecules import paths
from scf_guess_tools import Engine, Metric, PySCFEngine, Psi4Engine


def reproduce(reference: pd.DataFrame, basis: str, engine: Engine):
    table = pd.DataFrame(index=reference.index, columns=engine.guessing_schemes())
    molecules = {m.name: m for m in [engine.load(path) for path in paths()]}

    for name in table.index:
        for scheme in table.columns:
            molecule = molecules[name]
            initial = engine.guess(molecule, basis, scheme)
            final = engine.calculate(molecule, basis)
            table.at[name, scheme] = engine.score(initial, final, Metric.F_SCORE)

    return table


if __name__ == "__main__":
    pd.options.display.float_format = '{:.3f}'.format

    theories = ["HF"]
    bases = ["sto-3g"]
    variants = ["singlet", "non-singlet"]
    engines = [PySCFEngine(cache=False), Psi4Engine(cache=False)]

    for theory, basis, variant in product(theories, bases, variants):
        reference = load_table(theory, basis, variant).iloc[:1]
        print(f"{theory}/{basis}/{variant} reference:\n{reference}")

        for engine in [PySCFEngine, Psi4Engine]:
            reproduced = reproduce(reference, basis, engine(cache=False))
            print(f"{theory}/{basis}/{variant} via {engine.backend()}:\n{reproduced}")
