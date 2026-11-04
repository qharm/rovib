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

from .anharmonic_potential import AnharmonicPotential
from .coriolis import coriolis_couplings
from .driver import (
    dominant_basis_function,
    hartree_to_wavenumber,
    rovibrational_energies,
)
from .hamiltonian import Hamiltonian
from .hamiltonian_term import HamiltonianTerm
from .harmonic_hamiltonian import HarmonicHamiltonian
from .inertia import inertia_derivatives, inertia_tensor, rotational_constants
from .matrix_elements.hamiltonian_matrix import (
    build_hamiltonian_matrix,
    non_zero_indices,
)
from .molecule import Molecule
from .rigid_rotor_hamiltonian import RigidRotorHamiltonian
from .rovibrational_ansatz import RovibrationalAnsatz
from .rovibrational_basis_function import RovibrationalBasisFunction
from .vibrational_coriolis_hamiltonian import VibrationalCoriolisHamiltonian
from .water_params import (
    anharmonic_force_constants,
    harmonic_analysis,
    is_water,
    water_molecule,
)

__all__ = [
    "AnharmonicPotential",
    "Hamiltonian",
    "HamiltonianTerm",
    "HarmonicHamiltonian",
    "Molecule",
    "RigidRotorHamiltonian",
    "RovibrationalAnsatz",
    "RovibrationalBasisFunction",
    "VibrationalCoriolisHamiltonian",
    "anharmonic_force_constants",
    "build_hamiltonian_matrix",
    "coriolis_couplings",
    "dominant_basis_function",
    "harmonic_analysis",
    "hartree_to_wavenumber",
    "inertia_derivatives",
    "inertia_tensor",
    "is_water",
    "non_zero_indices",
    "rotational_constants",
    "rovibrational_energies",
    "water_molecule",
]
