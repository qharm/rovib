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

import unittest

import numpy as np

from rovib import RotationalAnsatz
from rovib.matrix_elements.rotational_matrix_elements import rotational_matrix
from rovib.operators.j_operator import JOperator


class TestRotationalMatrix(unittest.TestCase):
    def test_no_operators_is_identity(self):
        ansatz = RotationalAnsatz(2)
        matrix = rotational_matrix(ansatz, [], ansatz)
        self.assertEqual(matrix.dtype, np.complex128)
        np.testing.assert_array_equal(matrix, np.eye(5))

    def test_J_z(self):
        ansatz = RotationalAnsatz(2)
        matrix = rotational_matrix(ansatz, [JOperator(2, 2)], ansatz)
        np.testing.assert_array_equal(matrix, np.diag([-2, -1, 0, 1, 2]))

    def test_components_are_hermitian(self):
        ansatz = RotationalAnsatz(3)
        for projection in range(3):
            with self.subTest(projection=projection):
                op = JOperator(3, projection)
                matrix = rotational_matrix(ansatz, [op], ansatz)
                np.testing.assert_allclose(matrix, matrix.conj().T)

    def test_anomalous_commutator(self):
        # [J_alpha, J_beta] = -i J_gamma for cyclic alpha, beta, gamma.
        for J in range(5):
            ansatz = RotationalAnsatz(J)
            ops = [JOperator(J, a) for a in range(3)]
            for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                with self.subTest(J=J, a=a, b=b):
                    forward = rotational_matrix(
                        ansatz, [ops[a], ops[b]], ansatz
                    )
                    reverse = rotational_matrix(
                        ansatz, [ops[b], ops[a]], ansatz
                    )
                    expected = -1j * rotational_matrix(
                        ansatz, [ops[c]], ansatz
                    )
                    np.testing.assert_allclose(
                        forward - reverse, expected, atol=1.0e-12
                    )

    def test_J_squared(self):
        for J in range(6):
            with self.subTest(J=J):
                ansatz = RotationalAnsatz(J)
                total = sum(
                    rotational_matrix(ansatz, [JOperator(J, a)] * 2, ansatz)
                    for a in range(3)
                )
                np.testing.assert_allclose(
                    total, J * (J + 1) * np.eye(2 * J + 1), atol=1.0e-12
                )

    def test_product_is_matrix_product(self):
        # The operators conserve J, so no states are lost between factors.
        ansatz = RotationalAnsatz(2)
        ops = [JOperator(2, a) for a in (0, 1, 1, 0)]
        product = np.eye(5)
        for op in ops:
            product = product @ rotational_matrix(ansatz, [op], ansatz)
        np.testing.assert_allclose(
            rotational_matrix(ansatz, ops, ansatz), product, atol=1.0e-12
        )

    def test_different_J_is_zero(self):
        bras = RotationalAnsatz(1)
        kets = RotationalAnsatz(2)
        matrix = rotational_matrix(bras, [JOperator(2, 0)], kets)
        self.assertEqual(matrix.shape, (3, 5))
        np.testing.assert_array_equal(matrix, 0.0)

    def test_operator_with_wrong_J(self):
        ansatz = RotationalAnsatz(2)
        with self.assertRaises(ValueError):
            rotational_matrix(ansatz, [JOperator(1, 0)], ansatz)
