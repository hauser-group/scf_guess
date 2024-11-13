# SCF guess
[![Pytest](https://github.com/hauser-group/scf_guess/actions/workflows/pytest.yaml/badge.svg)](https://github.com/hauser-group/scf_guess/actions/workflows/pytest.yaml)
[![codecov](https://codecov.io/gh/hauser-group/scf_guess/graph/badge.svg?token=9R2NA1V5LU)](https://codecov.io/gh/hauser-group/scf_guess)

## Installation
Since the main dependency of this repository is `psi4`, we recommend using `conda` to setup the environment:
```
>>> conda create -n scf_guess psi4==1.9.1 -c conda-forge/label/libint_dev -c conda-forge
>>> conda activate scf_guess
>>> pip install -e .
```