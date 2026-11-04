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
evaluated with :func:`.vibrational_matrix`, and |HRR| is evaluated with
:func:`.rigid_rotor_matrix`.
"""

import numpy as np

from ..anharmonic_potential import AnharmonicPotential
from ..harmonic_hamiltonian import HarmonicHamiltonian
from ..rigid_rotor_hamiltonian import RigidRotorHamiltonian
from ..vibrational_coriolis_hamiltonian import VibrationalCoriolisHamiltonian
from .rigid_rotor_hamiltonian_elements import rigid_rotor_matrix
from .vibrational_matrix_elements import vibrational_matrix

#: The types of Hamiltonian terms that are sums of products of vibrational
#: operators. Their ``terms`` method returns the (operators, reducer) pairs.
_VIBRATIONAL_TERMS = (
    AnharmonicPotential,
    HarmonicHamiltonian,
    VibrationalCoriolisHamiltonian,
)


def _build_vibrational_matrix(bra_ansatz, term, ket_ansatz):
    """Builds the matrix of a sum of products of vibrational operators.

    The operators do not act on the rotational basis functions, so the matrix
    is the direct product of the matrix in the vibrational basis and the
    identity in the rotational basis.
    """
    for ansatz in (bra_ansatz, ket_ansatz):
        if ansatz.nvib != term.nvib:
            raise ValueError(
                f"The Hamiltonian has {term.nvib} modes, but the space has "
                f"{ansatz.nvib}."
            )
    if bra_ansatz.rotational != ket_ansatz.rotational:
        return np.zeros((len(bra_ansatz), len(ket_ansatz)), dtype=complex)
    bras = bra_ansatz.vibrational
    kets = ket_ansatz.vibrational
    matrix = np.zeros((len(bras), len(kets)), dtype=complex)
    for operators, reducer in term.terms():
        matrix += vibrational_matrix(bras, operators, kets, reducer)
    return np.kron(matrix, np.eye(len(ket_ansatz.rotational)))


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
    if type(term) is RigidRotorHamiltonian:
        return rigid_rotor_matrix(bra_ansatz, term, ket_ansatz)
    raise TypeError(
        f"Matrix elements of {type(term).__name__} are not implemented."
    )


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
