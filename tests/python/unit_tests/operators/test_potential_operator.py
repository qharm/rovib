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

from rovib.operators.normal_mode import NormalMode
from rovib.operators.potential_operator import PotentialOperator


class TestPotentialOperator(unittest.TestCase):
    def setUp(self):
        self.frequencies = [1.0, 2.0]
        cubic = np.zeros((2, 2, 2))
        cubic[0, 0, 0] = 6.0
        for idx in ((0, 0, 1), (0, 1, 0), (1, 0, 0)):
            cubic[idx] = 4.0
        self.operator = PotentialOperator(cubic, self.frequencies)

    def test_order(self):
        self.assertEqual(self.operator.order, 3)

    def test_vibrational_operators(self):
        q0 = NormalMode(0, 1.0)
        q1 = NormalMode(1, 2.0)
        products = list(self.operator.vibrational_operators())
        expected = [
            (q0, q0, q0),
            (q0, q0, q1),
            (q0, q1, q1),
            (q1, q1, q1),
        ]
        self.assertEqual(products, expected)

    def test_coefficient(self):
        q0 = NormalMode(0, 1.0)
        q1 = NormalMode(1, 2.0)
        # The force constant divided by the factorials of the powers.
        self.assertAlmostEqual(self.operator.coefficient((q0, q0, q0)), 1.0)
        self.assertAlmostEqual(self.operator.coefficient((q0, q0, q1)), 2.0)
        self.assertEqual(self.operator.coefficient((q1, q1, q1)), 0.0)

    def test_matches_unrestricted_sum(self):
        # The Q_i commute, so the operator can be checked as a polynomial:
        # sum_products c * prod x_n must equal
        # (1 / m!) * sum_{i_1...i_m} F_{i_1...i_m} x_{i_1} ... x_{i_m}.
        rng = np.random.default_rng(42)
        frequencies = [1.0, 2.0, 3.0]
        x = rng.uniform(-1.0, 1.0, 3)
        for order in (2, 3, 4):
            with self.subTest(order=order):
                tensor = rng.uniform(-1.0, 1.0, (3,) * order)
                permutations = list(itertools.permutations(range(order)))
                force_constants = sum(
                    tensor.transpose(p) for p in permutations
                ) / len(permutations)
                operator = PotentialOperator(force_constants, frequencies)
                value = sum(
                    operator.coefficient(ops)
                    * math.prod(x[op.mode] for op in ops)
                    for ops in operator.vibrational_operators()
                )
                expected = force_constants
                for _ in range(order):
                    expected = expected @ x
                self.assertAlmostEqual(value, expected / math.factorial(order))

    def test_wrong_shape(self):
        with self.assertRaises(ValueError):
            PotentialOperator(np.zeros((2, 3)), self.frequencies)
        with self.assertRaises(ValueError):
            PotentialOperator(np.zeros((3, 3)), self.frequencies)
        with self.assertRaises(ValueError):
            PotentialOperator(1.0, self.frequencies)
