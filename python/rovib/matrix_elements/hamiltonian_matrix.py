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

:func:`build_matrix` does not depend on the type of the term. Each term is a
sum of direct products of a product of vibrational operators and a product of
rotational operators, see :meth:`.HamiltonianTerm.terms`, so

.. math::

   \\braket{\\vvec',J',K'\\mid\\op{H}\\mid\\vvec,J,K} = \\sum_{t}
      \\braket{\\vvec'\\mid\\op{V}_t\\mid\\vvec}
      \\braket{J',K'\\mid\\op{R}_t\\mid J,K}

where the factors are evaluated with :func:`.vibrational_matrix` and
:func:`.rotational_matrix`.
"""

import numpy as np

from .rotational_matrix_elements import rotational_matrix
from .vibrational_matrix_elements import vibrational_matrix


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
    :raises ValueError: If ``term`` acts on the vibrational modes and either
        space has a different number of them than ``term``.
    """
    for ansatz in (bra_ansatz, ket_ansatz):
        if term.nvib is not None and ansatz.nvib != term.nvib:
            raise ValueError(
                f"The Hamiltonian has {term.nvib} modes, but the space has "
                f"{ansatz.nvib}."
            )
    matrix = np.zeros((len(bra_ansatz), len(ket_ansatz)), dtype=complex)
    for vibrational_ops, reducer, rotational_ops in term.terms(
        ket_ansatz.rotational
    ):
        matrix += np.kron(
            vibrational_matrix(
                bra_ansatz.vibrational,
                vibrational_ops,
                ket_ansatz.vibrational,
                reducer,
            ),
            rotational_matrix(
                bra_ansatz.rotational, rotational_ops, ket_ansatz.rotational
            ),
        )
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
