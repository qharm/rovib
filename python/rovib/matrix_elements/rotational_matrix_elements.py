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

"""Matrix elements of products of angular momentum operators."""

import numpy as np

from ..wavefunction.rotational_basis_function import RotationalBasisFunction
from .apply_operators import apply_operators


def rotational_matrix(bras, ops, kets):
    """Builds the matrix of a product of angular momentum operators.

    The operators conserve :math:`J`, so the matrix is zero unless ``bras``
    and ``kets`` have the same :math:`J`. An empty ``ops`` is the identity.

    :param bras: The space the bra basis functions come from.
    :type bras: RotationalAnsatz
    :param ops: The factors of the operator, in order.
    :type ops: list of JOperator
    :param kets: The space the ket basis functions come from.
    :type kets: RotationalAnsatz
    :return: The matrix of the operator in atomic units. The shape is
        (``len(bras)``, ``len(kets)``).
    :rtype: numpy.ndarray
    :raises ValueError: If any element of ``ops`` has a different :math:`J`
        than ``kets``.
    """
    for op in ops:
        if op.J != kets.J:
            raise ValueError(
                f"The operator has J={op.J}, but the kets have J={kets.J}."
            )
    matrix = np.zeros((len(bras), len(kets)), dtype=complex)
    if bras.J != kets.J:
        return matrix
    for ket in kets:
        k = kets.index(ket)
        for K, element in apply_operators(tuple(ops), ket.K).items():
            bra = RotationalBasisFunction(ket.J, K)
            matrix[bras.index(bra), k] = element
    return matrix
