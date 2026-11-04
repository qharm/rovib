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

import math
import unittest

import numpy as np

from rovib.operators.normal_mode import NormalMode
from rovib.operators.watson_operator import WatsonOperator


class TestWatsonOperator(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.frequencies = [1.0, 2.0]
        self.gamma = np.array([0.25, 1.0, 0.5])
        # N.b. 1 / I^0_alphaalpha = 2 gamma_alpha
        self.inverse_moments = 2.0 * self.gamma
        tensor = rng.uniform(-1.0, 1.0, (2, 3, 3))
        self.a = tensor + tensor.transpose(0, 2, 1)
        self.q = [NormalMode(0, 1.0), NormalMode(1, 2.0)]

    def _operator(self, order):
        return WatsonOperator(order, self.a, self.gamma, self.frequencies)

    def test_order(self):
        self.assertEqual(self._operator(3).order, 3)

    def test_vibrational_operators(self):
        q0, q1 = self.q
        self.assertEqual(list(self._operator(0).vibrational_operators()), [()])
        self.assertEqual(
            list(self._operator(1).vibrational_operators()),
            [(q0,), (q1,)],
        )
        self.assertEqual(
            list(self._operator(2).vibrational_operators()),
            [(q0, q0), (q0, q1), (q1, q0), (q1, q1)],
        )

    def test_order_0(self):
        eta = self._operator(0).coefficient(())
        np.testing.assert_allclose(eta, np.diag(self.inverse_moments))

    def test_order_1(self):
        eta = self._operator(1).coefficient((self.q[1],))
        inverse = np.diag(self.inverse_moments)
        np.testing.assert_allclose(eta, -inverse @ self.a[1] @ inverse)

    def test_order_2(self):
        eta = self._operator(2).coefficient((self.q[0], self.q[1]))
        inverse = np.diag(self.inverse_moments)
        np.testing.assert_allclose(
            eta, 0.75 * inverse @ self.a[0] @ inverse @ self.a[1] @ inverse
        )

    def test_order_4_sign(self):
        # c_4 = (4 + 1) * (-1 / 2)^4 = 5 / 16
        eta = self._operator(4).coefficient((self.q[0],) * 4)
        inverse = np.diag(self.inverse_moments)
        product = inverse
        for _ in range(4):
            product = product @ self.a[0] @ inverse
        np.testing.assert_allclose(eta, 0.3125 * product)

    def test_is_taylor_series_of_inverse(self):
        # With I' = (I^0 + a.x / 2) (I^0)^-1 (I^0 + a.x / 2), the truncation
        # error of the sum of the orders must be O(x^5).
        direction = np.array([0.6, -0.8])
        moments = np.diag(1.0 / self.inverse_moments)
        inverse = np.diag(self.inverse_moments)

        def error(scale):
            x = scale * direction
            half = moments + 0.5 * np.einsum("kab,k->ab", self.a, x)
            exact = np.linalg.inv(half @ inverse @ half)
            total = np.zeros((3, 3))
            for order in range(5):
                operator = self._operator(order)
                for ops in operator.vibrational_operators():
                    total = total + operator.coefficient(ops) * math.prod(
                        x[op.mode] for op in ops
                    )
            return np.abs(total - exact).max()

        ratio = error(0.02) / error(0.01)
        self.assertAlmostEqual(math.log2(ratio), 5.0, delta=0.1)

    def test_coefficient_is_read_only(self):
        eta = self._operator(1).coefficient((self.q[0],))
        with self.assertRaises(ValueError):
            eta[0, 0] = 1.0

    def test_negative_order(self):
        with self.assertRaises(ValueError):
            self._operator(-1)

    def test_wrong_shape(self):
        for shape in ((3, 3), (2, 3, 2), (3, 3, 3)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    WatsonOperator(
                        1, np.zeros(shape), self.gamma, self.frequencies
                    )
        with self.assertRaises(ValueError):
            WatsonOperator(1, self.a, [1.0, 2.0], self.frequencies)
