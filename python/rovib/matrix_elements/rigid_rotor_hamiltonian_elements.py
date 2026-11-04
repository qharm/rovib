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

"""Matrix of the rigid rotor Hamiltonian.

In atomic units,

.. math::

   \\HRR = \\sum_{\\alpha}\\gamma_\\alpha\\jop_\\alpha^2

so |HRR| is the direct product of the identity in the vibrational basis and a
sum of products of :class:`.JOperator` objects in the rotational basis. See
:ref:`rigid_rotor` for the closed form of the matrix elements.
"""

import numpy as np

from .rotational_matrix_elements import rotational_matrix
from .vibrational_matrix_elements import vibrational_matrix


def _unit_coefficient(coefficients):
    """Reducer for the identity, i.e., a product of no operators."""
    return 1.0


def rigid_rotor_matrix(bra_ansatz, term, ket_ansatz):
    """Builds the dense matrix of |HRR|.

    The bra and ket spaces may differ, e.g., to build a block of the matrix.

    :param bra_ansatz: The space the bra basis functions come from.
    :type bra_ansatz: RovibrationalAnsatz
    :param term: The rigid rotor Hamiltonian.
    :type term: RigidRotorHamiltonian
    :param ket_ansatz: The space the ket basis functions come from.
    :type ket_ansatz: RovibrationalAnsatz
    :return: The matrix of ``term`` in hartree. The shape is
        (``len(bra_ansatz)``, ``len(ket_ansatz)``).
    :rtype: numpy.ndarray
    """
    # N.b. the spaces may number the vibrational basis functions differently,
    # so the overlap is not, in general, an identity matrix.
    overlap = vibrational_matrix(
        bra_ansatz.vibrational, [], ket_ansatz.vibrational, _unit_coefficient
    )
    bras = bra_ansatz.rotational
    kets = ket_ansatz.rotational
    rotational = np.zeros((len(bras), len(kets)), dtype=complex)
    products = term.rotational_operators(kets)
    for gamma, ops in zip(term.rotational_constants, products):
        rotational += gamma * rotational_matrix(bras, ops, kets)
    return np.kron(overlap, rotational)
