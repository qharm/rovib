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

import dataclasses
import math
import unittest

from rovib.operators.j_operator import JOperator


class TestJOperator(unittest.TestCase):
    def test_state(self):
        op = JOperator(2, 1)
        self.assertEqual(op.J, 2)
        self.assertEqual(op.projection, 1)

    def test_act_z(self):
        self.assertEqual(JOperator(3, 2).act(-2), [(-2, -2.0)])

    def test_act_x(self):
        result = dict(JOperator(2, 0).act(1))
        self.assertEqual(set(result), {0, 2})
        # (C^-_{2,1} / 2, C^+_{2,1} / 2)
        self.assertAlmostEqual(result[0], 0.5 * math.sqrt(6.0))
        self.assertAlmostEqual(result[2], 0.5 * 2.0)

    def test_act_y(self):
        result = dict(JOperator(2, 1).act(1))
        self.assertEqual(set(result), {0, 2})
        # J_+ / (2i) lowers K and -J_- / (2i) raises it.
        self.assertAlmostEqual(result[0], -0.5j * math.sqrt(6.0))
        self.assertAlmostEqual(result[2], 0.5j * 2.0)

    def test_ladder_operators_are_reversed(self):
        # J_x + i J_y = J_+ must only lower K in the molecule-fixed frame.
        x = dict(JOperator(2, 0).act(1))
        y = dict(JOperator(2, 1).act(1))
        self.assertAlmostEqual(x[2] + 1j * y[2], 0.0)
        self.assertAlmostEqual(x[0] + 1j * y[0], math.sqrt(6.0))

    def test_act_stays_in_range(self):
        for projection in (0, 1):
            with self.subTest(projection=projection):
                op = JOperator(1, projection)
                self.assertEqual([K for K, _ in op.act(1)], [0])
                self.assertEqual([K for K, _ in op.act(-1)], [0])
                self.assertEqual(JOperator(0, projection).act(0), [])

    def test_immutable(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            JOperator(2, 1).J = 3

    def test_hashable(self):
        self.assertEqual(hash(JOperator(2, 1)), hash(JOperator(2, 1)))
        self.assertNotEqual(JOperator(2, 1), JOperator(2, 0))

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            JOperator(-1, 0)
        for projection in (-1, 3):
            with self.subTest(projection=projection):
                with self.assertRaises(ValueError):
                    JOperator(1, projection)
