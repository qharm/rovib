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

"""Matrix elements of products of vibrational operators.

Implements "Vibrational Basis: Algorithm 1" from :ref:`matrix_elements`.
Operators on different modes commute, so a matrix element of a product of
single-mode operators factors into one matrix element per mode,

.. math::

   \\braket{\\vvec'\\mid\\op{O}_1\\cdots\\op{O}_m\\mid\\vvec} =
      \\prod_{n=1}^{\\Nvib}\\braket{v'_n\\mid\\op{O}^{(n)}\\mid v_n}

where :math:`\\op{O}^{(n)}` is the product, in the original order, of the
operators acting on mode :math:`n` (the identity if there are none).
"""

import collections
import functools
import itertools
import math

import numpy as np

from ..wavefunction.vibrational_basis_function import VibrationalBasisFunction
from .apply_operators import apply_operators


# N.b. operators are immutable, so the result can be cached. This matters
# because it is needed for every ket.
@functools.cache
def _products(operators, reducer):
    """The non-zero terms of the product of ``operators``.

    :return: A tuple of (coefficient, words) pairs, where words is a tuple of
        (mode, operators acting on that mode) pairs.
    """
    products = []
    ranges = [o.vibrational_operators() for o in operators]
    for op_lists in itertools.product(*ranges):
        coefficients = [o.coefficient(x) for o, x in zip(operators, op_lists)]
        coefficient = reducer(coefficients)
        if coefficient == 0:
            continue
        words = collections.defaultdict(list)
        for operator in itertools.chain.from_iterable(op_lists):
            words[operator.mode].append(operator)
        words = tuple((n, tuple(word)) for n, word in words.items())
        products.append((coefficient, words))
    return tuple(products)


def vibrational_matrix(bras, ops, kets, reducer):
    """Builds the matrix of a product of vibrational operators.

    ``ops`` is a list of objects that each represent a sum of products of
    single-mode operators, e.g., :class:`.PotentialOperator`. The operator
    whose matrix is built is the product of the elements of ``ops``, in order,
    with each term's coefficient given by ``reducer``. An empty ``ops`` is the
    identity, scaled by ``reducer([])``.

    :param bras: The space the bra basis functions come from.
    :type bras: VibrationalAnsatz
    :param ops: The factors of the operator.
    :type ops: list
    :param kets: The space the ket basis functions come from.
    :type kets: VibrationalAnsatz
    :param reducer: Combines a list holding the coefficient of one product
        from each element of ``ops`` into the coefficient of their product.
        Products whose coefficient is zero are skipped.
    :type reducer: collections.abc.Callable
    :return: The matrix of the operator in atomic units. The shape is
        (``len(bras)``, ``len(kets)``).
    :rtype: numpy.ndarray
    """
    matrix = np.zeros((len(bras), len(kets)), dtype=complex)
    products = _products(tuple(ops), reducer)
    for ket in kets:
        k = kets.index(ket)
        # Different products can couple the same bra and ket, so their
        # contributions are summed.
        for coefficient, words in products:
            per_mode = [
                apply_operators(word, ket.v[n]).items() for n, word in words
            ]
            for elements in itertools.product(*per_mode):
                v = list(ket.v)
                for (n, _), (v_n, _) in zip(words, elements):
                    v[n] = v_n
                bra = VibrationalBasisFunction(tuple(v))
                if bra not in bras:
                    continue
                value = coefficient * math.prod(e for _, e in elements)
                matrix[bras.index(bra), k] += value
    return matrix
