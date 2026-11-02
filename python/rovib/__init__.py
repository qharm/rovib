# Copyright 2026 QHARM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A proof-of-concept library that solves the rovibrational problem."""

from .driver import (
    dominant_basis_function,
    encode_harmonic_hamiltonian,
    encoded_harmonic_hamiltonian,
    hartree_to_wavenumber,
    print_encoded_harmonic_hamiltonian,
    rovibrational_energies,
)
from .hamiltonian import Hamiltonian
from .hamiltonian_matrix import (
    build_hamiltonian_matrix,
    matrix_element,
    non_zero_indices,
)
from .hamiltonian_term import HamiltonianTerm
from .harmonic_hamiltonian import HarmonicHamiltonian
from .molecule import Molecule
from .rovibrational_ansatz import RovibrationalAnsatz
from .rovibrational_basis_function import RovibrationalBasisFunction
from .water_params import harmonic_analysis, is_water, water_molecule

__all__ = [
    "Hamiltonian",
    "HamiltonianTerm",
    "HarmonicHamiltonian",
    "Molecule",
    "RovibrationalAnsatz",
    "RovibrationalBasisFunction",
    "build_hamiltonian_matrix",
    "dominant_basis_function",
    "encode_harmonic_hamiltonian",
    "encoded_harmonic_hamiltonian",
    "harmonic_analysis",
    "hartree_to_wavenumber",
    "is_water",
    "matrix_element",
    "non_zero_indices",
    "print_encoded_harmonic_hamiltonian",
    "rovibrational_energies",
    "water_molecule",
]
