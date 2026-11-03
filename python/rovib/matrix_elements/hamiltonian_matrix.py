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

:func:`build_matrix` dispatches on the type of the term. Terms that are
products of vibrational operators (e.g., :class:`.AnharmonicPotential`) are
evaluated with :func:`.vibrational_matrix`. For the remaining terms, the
elements returned by :func:`non_zero_indices` are evaluated one at a time
with the term's matrix element function.
"""

import numpy as np

from ..anharmonic_potential import AnharmonicPotential
from ..harmonic_hamiltonian import HarmonicHamiltonian
from ..rigid_rotor_hamiltonian import RigidRotorHamiltonian
from .harmonic_hamiltonian_elements import (
    harmonic_matrix_element,
    harmonic_non_zero_indices,
)
from .rigid_rotor_hamiltonian_elements import (
    rigid_rotor_matrix_element,
    rigid_rotor_non_zero_indices,
)
from .vibrational_matrix_elements import vibrational_matrix

#: The types of Hamiltonian terms that are sums of products of vibrational
#: operators. Their ``terms`` method returns the (operators, reducer) pairs.
_VIBRATIONAL_TERMS = (AnharmonicPotential,)

#: Maps each type of Hamiltonian term to its non-zero index generator.
_NON_ZERO_INDICES = {
    HarmonicHamiltonian: harmonic_non_zero_indices,
    RigidRotorHamiltonian: rigid_rotor_non_zero_indices,
}

#: Maps each type of Hamiltonian term to its matrix element function.
_MATRIX_ELEMENTS = {
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
    matrix. Terms that are products of vibrational operators are not
    supported, see :func:`build_matrix`.

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


def _build_vibrational_matrix(bra_ansatz, term, ket_ansatz):
    """Builds the matrix of a sum of products of vibrational operators."""
    for ansatz in (bra_ansatz, ket_ansatz):
        if ansatz.nvib != term.nvib:
            raise ValueError(
                f"The Hamiltonian has {term.nvib} modes, but the space has "
                f"{ansatz.nvib}."
            )
    matrix = np.zeros((len(bra_ansatz), len(ket_ansatz)), dtype=complex)
    for operators, reducer in term.terms():
        matrix += vibrational_matrix(
            bra_ansatz, ket_ansatz, operators, reducer
        )
    return matrix


def build_matrix(bra_ansatz, term, ket_ansatz):
    """Builds the dense matrix of a single Hamiltonian term.

    The bra and ket spaces may differ, e.g., to build a block of the matrix.

    :param bra_ansatz: The space the bra basis functions come from.
    :type bra_ansatz: RovibrationalAnsatz
    :param term: The Hamiltonian term.
    :type term: HamiltonianTerm
    :param ket_ansatz: The space the ket basis functions come from.
    :type ket_ansatz: RovibrationalAnsatz
    :return: The matrix of ``term`` in hartree. The shape is
        (``len(bra_ansatz)``, ``len(ket_ansatz)``).
    :rtype: numpy.ndarray
    :raises TypeError: If the type of ``term`` is not implemented.
    :raises ValueError: If either space has a different number of
        vibrational modes than ``term``.
    """
    if type(term) in _VIBRATIONAL_TERMS:
        return _build_vibrational_matrix(bra_ansatz, term, ket_ansatz)
    matrix = np.zeros((len(bra_ansatz), len(ket_ansatz)), dtype=complex)
    for bra, ket in non_zero_indices(bra_ansatz, term, ket_ansatz):
        b = bra_ansatz.index(bra)
        k = ket_ansatz.index(ket)
        matrix[b, k] += _implementation(_MATRIX_ELEMENTS, term)(bra, term, ket)
    return matrix


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
        matrix += build_matrix(ansatz, term, ansatz)

    if not np.allclose(matrix, matrix.conj().T):
        raise RuntimeError("The Hamiltonian matrix is not Hermitian.")
    return matrix
