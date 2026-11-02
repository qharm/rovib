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

"""Builds the matrix of a Hamiltonian in a rovibrational ansatz.

:func:`non_zero_indices` and :func:`matrix_element` are the common APIs for
all Hamiltonian terms; internally they dispatch on the type of the term.
:func:`build_hamiltonian_matrix` uses them to fill in the non-zero elements
of a dense matrix.
"""

import numpy as np

from .anharmonic_potential import AnharmonicPotential
from .anharmonic_potential_elements import (
    anharmonic_matrix_element,
    anharmonic_non_zero_indices,
)
from .harmonic_hamiltonian import HarmonicHamiltonian
from .harmonic_hamiltonian_elements import (
    harmonic_matrix_element,
    harmonic_non_zero_indices,
)
from .rigid_rotor_hamiltonian import RigidRotorHamiltonian
from .rigid_rotor_hamiltonian_elements import (
    rigid_rotor_matrix_element,
    rigid_rotor_non_zero_indices,
)

#: Maps each type of Hamiltonian term to its non-zero index generator.
_NON_ZERO_INDICES = {
    AnharmonicPotential: anharmonic_non_zero_indices,
    HarmonicHamiltonian: harmonic_non_zero_indices,
    RigidRotorHamiltonian: rigid_rotor_non_zero_indices,
}

#: Maps each type of Hamiltonian term to its matrix element function.
_MATRIX_ELEMENTS = {
    AnharmonicPotential: anharmonic_matrix_element,
    HarmonicHamiltonian: harmonic_matrix_element,
    RigidRotorHamiltonian: rigid_rotor_matrix_element,
}


def _implementation(table, term):
    """Looks up the implementation for the type of ``term`` in ``table``."""
    try:
        return table[type(term)]
    except KeyError:
        raise TypeError(
            f"Matrix elements of {type(term).__name__} are not implemented."
        ) from None


def non_zero_indices(bra_space, term, ket_space):
    """Generates the bra-ket pairs whose matrix elements may be non-zero.

    Pairs whose matrix element is zero by symmetry or selection rules are
    skipped. The bra and ket spaces may differ, e.g., to build a block of the
    matrix.

    :param bra_space: The space the bra basis functions come from.
    :type bra_space: RovibrationalAnsatz
    :param term: The Hamiltonian term.
    :type term: HamiltonianTerm
    :param ket_space: The space the ket basis functions come from.
    :type ket_space: RovibrationalAnsatz
    :return: A generator of (bra, ket) pairs of
        :class:`.RovibrationalBasisFunction` objects.
    :rtype: collections.abc.Iterator
    :raises TypeError: If the type of ``term`` is not implemented.
    """
    generator = _implementation(_NON_ZERO_INDICES, term)
    return generator(bra_space, term, ket_space)


def matrix_element(bra, term, ket):
    """Computes a single matrix element of a Hamiltonian term.

    :param bra: The basis function in the bra.
    :type bra: RovibrationalBasisFunction
    :param term: The Hamiltonian term.
    :type term: HamiltonianTerm
    :param ket: The basis function in the ket.
    :type ket: RovibrationalBasisFunction
    :return: The matrix element in hartree.
    :rtype: complex
    :raises TypeError: If the type of ``term`` is not implemented.
    """
    return _implementation(_MATRIX_ELEMENTS, term)(bra, term, ket)


def build_hamiltonian_matrix(hamiltonian, ansatz):
    """Builds the dense matrix of a Hamiltonian.

    :param hamiltonian: The Hamiltonian.
    :type hamiltonian: Hamiltonian
    :param ansatz: The space the Hamiltonian is built in.
    :type ansatz: RovibrationalAnsatz
    :return: The matrix of ``hamiltonian`` in hartree. The shape is
        (``len(ansatz)``, ``len(ansatz)``).
    :rtype: numpy.ndarray
    :raises TypeError: If ``hamiltonian`` contains a term whose matrix
        elements are not implemented.
    :raises RuntimeError: If the resulting matrix is not Hermitian.
    """
    size = len(ansatz)
    matrix = np.zeros((size, size), dtype=complex)
    for term in hamiltonian:
        for bra, ket in non_zero_indices(ansatz, term, ansatz):
            b = ansatz.index(bra)
            k = ansatz.index(ket)
            matrix[b, k] += matrix_element(bra, term, ket)
    if not np.allclose(matrix, matrix.conj().T):
        raise RuntimeError("The Hamiltonian matrix is not Hermitian.")
    return matrix
