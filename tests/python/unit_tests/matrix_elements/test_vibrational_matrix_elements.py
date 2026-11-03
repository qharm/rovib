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

import itertools
import math
import unittest

import numpy as np

from rovib import RovibrationalAnsatz
from rovib.matrix_elements.vibrational_matrix_elements import (
    vibrational_matrix,
)
from rovib.operators.conjugate_momentum import ConjugateMomentum
from rovib.operators.normal_mode import NormalMode


class _ProductOperator:
    """A sum of products of single-mode operators with given coefficients.

    Stands in for the CoriolisOperator and WatsonOperator classes.
    """

    def __init__(self, coefficients):
        self._coefficients = dict(coefficients)

    def vibrational_operators(self):
        return self._coefficients.keys()

    def coefficient(self, operators):
        return self._coefficients[operators]


def _product(coefficients):
    return math.prod(coefficients)


def _brute_force_matrix(operators, frequencies, vmax, max_quanta):
    """The matrix of the product of ``operators`` in a direct product basis.

    Each single-mode operator is represented in a basis that is large enough
    for products of up to ``max_quanta`` operators to be exact within the
    returned block, which has :math:`0 \\le v_n \\le v_{\\max}`.
    """
    nvib = len(frequencies)
    dim = vmax + max_quanta + 1
    ladder = np.diag(np.sqrt(np.arange(1, dim)), 1)
    identity = np.eye(dim)

    def single(operator):
        omega = frequencies[operator.mode]
        if isinstance(operator, NormalMode):
            matrix = np.sqrt(0.5 / omega) * (ladder + ladder.T)
        else:
            matrix = 1j * np.sqrt(0.5 * omega) * (ladder.T - ladder)
        factors = [identity] * nvib
        factors[operator.mode] = matrix
        result = factors[0]
        for factor in factors[1:]:
            result = np.kron(result, factor)
        return result

    total = 0.0
    ranges = [o.vibrational_operators() for o in operators]
    for op_lists in itertools.product(*ranges):
        term = _product(
            o.coefficient(x) for o, x in zip(operators, op_lists)
        ) * np.eye(dim**nvib)
        for operator in itertools.chain.from_iterable(op_lists):
            term = term @ single(operator)
        total = total + term

    keep = [
        np.ravel_multi_index(idx, (dim,) * nvib)
        for idx in itertools.product(range(vmax + 1), repeat=nvib)
    ]
    return total[np.ix_(keep, keep)]


class TestVibrationalMatrix(unittest.TestCase):
    def setUp(self):
        self.frequencies = [1.0, 2.0]
        q = [NormalMode(n, w) for n, w in enumerate(self.frequencies)]
        p = [ConjugateMomentum(n, w) for n, w in enumerate(self.frequencies)]
        # A Coriolis-like factor, sum_ij c_ij Q_i P_j, ...
        self.coriolis = _ProductOperator(
            {
                (q[0], p[1]): 0.7,
                (q[1], p[0]): -0.7,
                (q[0], p[0]): 0.3,
            }
        )
        # ... and a Watson-like factor, sum_k c_k Q_k.
        self.watson = _ProductOperator({(q[0],): 1.5, (q[1],): -0.5})
        self.operators = [self.coriolis, self.watson, self.coriolis]
        self.ansatz = RovibrationalAnsatz(2, 3, 0)

    def test_matches_brute_force(self):
        matrix = vibrational_matrix(
            self.ansatz, self.ansatz, self.operators, _product
        )
        self.assertEqual(matrix.dtype, np.complex128)
        np.testing.assert_allclose(
            matrix,
            _brute_force_matrix(self.operators, self.frequencies, 3, 5),
            atol=1.0e-12,
        )

    def test_matches_brute_force_imaginary(self):
        # A single momentum per term makes the elements imaginary.
        matrix = vibrational_matrix(
            self.ansatz, self.ansatz, [self.coriolis], _product
        )
        self.assertGreater(np.abs(matrix.imag).max(), 0.0)
        np.testing.assert_allclose(
            matrix,
            _brute_force_matrix([self.coriolis], self.frequencies, 3, 2),
            atol=1.0e-12,
        )

    def test_diagonal_in_J_and_K(self):
        ansatz = RovibrationalAnsatz(2, 2, 1)
        matrix = vibrational_matrix(ansatz, ansatz, [self.watson], _product)
        self.assertGreater(np.count_nonzero(matrix), 0)
        for b, k in zip(*np.nonzero(matrix)):
            bra = ansatz.basis_function(b)
            ket = ansatz.basis_function(k)
            self.assertEqual((bra.J, bra.K), (ket.J, ket.K))

    def test_zero_coefficients_skipped(self):
        zero = _ProductOperator({(NormalMode(0, 1.0),): 0.0})
        matrix = vibrational_matrix(self.ansatz, self.ansatz, [zero], _product)
        np.testing.assert_array_equal(matrix, 0.0)

    def test_different_spaces(self):
        bra_space = RovibrationalAnsatz(2, 1, 0)
        block = vibrational_matrix(
            bra_space, self.ansatz, self.operators, _product
        )
        self.assertEqual(block.shape, (len(bra_space), len(self.ansatz)))
        full = vibrational_matrix(
            self.ansatz, self.ansatz, self.operators, _product
        )
        rows = [self.ansatz.index(bra) for bra in bra_space]
        np.testing.assert_allclose(block, full[rows, :], atol=1.0e-12)
