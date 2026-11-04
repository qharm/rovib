<!--
  ~ Copyright 2026 QHARM Team
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~ http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
-->

# rovib
A proof-of-concept library that solves the rovibrational problem

> **Note:** rovib is under active development. The Hamiltonian contains the
> harmonic oscillator, rigid rotor, anharmonic, vibrational Coriolis, and
> rovibrational coupling terms. The harmonic and anharmonic parameters are
> hard-coded for water (see `rovib.water_params`).

## Installation

rovib requires Python 3.12 or newer and NumPy. To install it from a clone of
the repository:

```sh
git clone https://github.com/qharm/rovib.git
cd rovib
pip install .
```

Developers should install rovib in editable mode instead, so that changes to
the source are picked up without reinstalling:

```sh
pip install -e .
```

## Usage

The driver, `rovibrational_energies`, takes a molecule, the maximum
vibrational quantum number of each mode (`vmax`), and the total angular
momentum quantum number (`J`). It returns the energies in hartree and the
corresponding eigenvectors. The following computes the J = 0 levels of water:

```python
from rovib import (
    RovibrationalAnsatz,
    dominant_basis_function,
    hartree_to_wavenumber,
    rovibrational_energies,
    water_molecule,
)

water = water_molecule()
vmax, J = 3, 0
energies, eigenvectors = rovibrational_energies(water, vmax, J)

# The eigenvectors are expressed in this basis
ansatz = RovibrationalAnsatz(water.nvib, vmax, J)

wavenumbers = hartree_to_wavenumber(energies)
print(f"Zero-point energy: {wavenumbers[0]:.1f} cm^-1")
for i in range(1, 5):
    bf = dominant_basis_function(eigenvectors[:, i], ansatz)
    print(bf.v, f"{wavenumbers[i] - wavenumbers[0]:.1f} cm^-1")
```

which prints:

```
Zero-point energy: 4641.5 cm^-1
(0, 1, 0) 1590.2 cm^-1
(0, 2, 0) 3162.0 cm^-1
(1, 0, 0) 3715.8 cm^-1
(0, 0, 1) 3797.0 cm^-1
```

The tuples are the vibrational quantum numbers (v1, v2, v3) of the basis
function that dominates each eigenvector.

## Development

### Testing

The unit and system tests use `unittest`:

```sh
python -m unittest discover tests/python
```

### Formatting and linting

Formatting and linting are enforced with [pre-commit](https://pre-commit.com).
To run the hooks automatically on each commit:

```sh
pip install pre-commit
pre-commit install
```
