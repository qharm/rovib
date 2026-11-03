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
"""The top-level driver: from a molecule to rovibrational energies."""

import numpy as np

from .constants import HARTREE_TO_CM
from .hamiltonian import Hamiltonian
from .harmonic_hamiltonian import HarmonicHamiltonian
from .inertia import rotational_constants
from .matrix_elements.hamiltonian_matrix import build_hamiltonian_matrix
from .rigid_rotor_hamiltonian import RigidRotorHamiltonian
from .rovibrational_ansatz import RovibrationalAnsatz
from .water_params import harmonic_analysis


def rovibrational_energies(molecule, vmax, J=0):
    """Computes the rovibrational energies of a molecule.

    The Hamiltonian is currently the harmonic oscillator Hamiltonian |HHO|
    plus the rigid rotor Hamiltonian |HRR|. The harmonic parameters are
    hard-coded for water. The eigenvectors are expressed in
    ``RovibrationalAnsatz(molecule.nvib, vmax, J)``.

    :param molecule: The molecule, at its minimum energy geometry, with its
        center of mass at the origin.
    :type molecule: Molecule
    :param vmax: The maximum vibrational quantum number of each mode.
    :type vmax: int
    :param J: The total angular momentum quantum number.
    :type J: int
    :return: The energies in hartree (in ascending order) and the
        corresponding eigenvectors (as columns).
    :rtype: tuple of (numpy.ndarray, numpy.ndarray)
    :raises ValueError: If ``molecule`` is not water.
    """
    # The normal modes are not needed until the Coriolis terms are added.
    _, frequencies = harmonic_analysis(molecule)

    H_harm = HarmonicHamiltonian(frequencies)
    H_rr = RigidRotorHamiltonian(rotational_constants(molecule))

    hamiltonian = Hamiltonian((H_harm, H_rr))
    ansatz = RovibrationalAnsatz(molecule.nvib, vmax, J)
    matrix = build_hamiltonian_matrix(hamiltonian, ansatz)
    energies, eigenvectors = np.linalg.eigh(matrix)
    return energies, eigenvectors


def dominant_basis_function(eigenvector, ansatz):
    """Returns the basis function that contributes most to an eigenvector.

    :param eigenvector: The eigenvector's coefficients in ``ansatz``.
    :type eigenvector: numpy.ndarray
    :param ansatz: The ansatz the eigenvector is expressed in.
    :type ansatz: RovibrationalAnsatz
    :return: The basis function with the largest coefficient magnitude.
    :rtype: RovibrationalBasisFunction
    """

    # N.b. np.argmax(np.abs(...)) is the index of the largest absolute value.
    return ansatz.basis_function(int(np.argmax(np.abs(eigenvector))))


def hartree_to_wavenumber(energy):
    """Converts energies from hartree to wavenumbers.

    :param energy: The energies in hartree.
    :type energy: numpy.ndarray
    :return: The energies in cm^-1.
    :rtype: numpy.ndarray
    """
    return np.asarray(energy) * HARTREE_TO_CM
