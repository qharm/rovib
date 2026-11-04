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

from rovib.operators.conjugate_momentum import ConjugateMomentum
from rovib.operators.coriolis_operator import CoriolisOperator
from rovib.operators.normal_mode import NormalMode


class TestCoriolisOperator(unittest.TestCase):
    def setUp(self):
        self.frequencies = [1.0, 2.0]
        self.zeta = np.zeros((2, 3, 2))
        self.zeta[0, :, 1] = [1.0, 2.0, 3.0]
        self.zeta[1, :, 0] = [-1.0, -2.0, -3.0]
        self.operator = CoriolisOperator(self.zeta, self.frequencies)
        self.q = [NormalMode(0, 1.0), NormalMode(1, 2.0)]
        self.p = [ConjugateMomentum(0, 1.0), ConjugateMomentum(1, 2.0)]

    def test_vibrational_operators(self):
        expected = [
            (self.q[0], self.p[0]),
            (self.q[0], self.p[1]),
            (self.q[1], self.p[0]),
            (self.q[1], self.p[1]),
        ]
        self.assertEqual(list(self.operator.vibrational_operators()), expected)

    def test_coefficient(self):
        for i in range(2):
            for j in range(2):
                with self.subTest(i=i, j=j):
                    np.testing.assert_array_equal(
                        self.operator.coefficient((self.q[i], self.p[j])),
                        self.zeta[i, :, j],
                    )

    def test_coefficient_is_read_only(self):
        coefficient = self.operator.coefficient((self.q[0], self.p[1]))
        with self.assertRaises(ValueError):
            coefficient[0] = 100.0

    def test_input_is_copied(self):
        self.zeta[0, 0, 1] = 100.0
        coefficient = self.operator.coefficient((self.q[0], self.p[1]))
        self.assertEqual(coefficient[0], 1.0)

    def test_wrong_shape(self):
        for shape in ((2, 2), (2, 2, 2), (3, 3, 3), (2, 3, 3)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    CoriolisOperator(np.zeros(shape), self.frequencies)
