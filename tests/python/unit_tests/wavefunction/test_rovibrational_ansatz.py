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

from rovib import (
    RotationalAnsatz,
    RovibrationalAnsatz,
    RovibrationalBasisFunction,
    VibrationalAnsatz,
)


class TestRovibrationalAnsatz(unittest.TestCase):
    def setUp(self):
        self.ansatz = RovibrationalAnsatz(3, 2, 1)

    def test_state(self):
        self.assertEqual(self.ansatz.nvib, 3)
        self.assertEqual(self.ansatz.vmax, 2)
        self.assertEqual(self.ansatz.J, 1)
        self.assertEqual(self.ansatz.vibrational, VibrationalAnsatz(3, 2))
        self.assertEqual(self.ansatz.rotational, RotationalAnsatz(1))

    def test_len(self):
        self.assertEqual(len(self.ansatz), 3**3 * 3)
        self.assertEqual(len(RovibrationalAnsatz(3, 0, 0)), 1)
        self.assertEqual(len(RovibrationalAnsatz(2, 4, 2)), 5**2 * 5)

    def test_iteration_order(self):
        functions = list(RovibrationalAnsatz(2, 1, 1))
        expected = [
            RovibrationalBasisFunction(v, 1, K)
            for v in [(0, 0), (0, 1), (1, 0), (1, 1)]
            for K in (-1, 0, 1)
        ]
        self.assertEqual(functions, expected)

    def test_iteration_matches_len(self):
        functions = list(self.ansatz)
        self.assertEqual(len(functions), len(self.ansatz))
        self.assertEqual(len(set(functions)), len(self.ansatz))

    def test_basis_function_and_index_round_trip(self):
        for i, bf in enumerate(self.ansatz):
            self.assertEqual(self.ansatz.basis_function(i), bf)
            self.assertEqual(self.ansatz.index(bf), i)

    def test_basis_function_out_of_range(self):
        with self.assertRaises(IndexError):
            self.ansatz.basis_function(-1)
        with self.assertRaises(IndexError):
            self.ansatz.basis_function(len(self.ansatz))

    def test_contains(self):
        self.assertIn(
            RovibrationalBasisFunction((2, 0, 2), 1, -1), self.ansatz
        )
        # vmax exceeded
        self.assertNotIn(
            RovibrationalBasisFunction((3, 0, 0), 1, 0), self.ansatz
        )
        # Wrong J
        self.assertNotIn(
            RovibrationalBasisFunction((0, 0, 0), 2, 0), self.ansatz
        )
        # Wrong number of modes
        self.assertNotIn(RovibrationalBasisFunction((0, 0), 1, 0), self.ansatz)
        self.assertNotIn("not a basis function", self.ansatz)

    def test_index_not_in_ansatz(self):
        with self.assertRaises(ValueError):
            self.ansatz.index(RovibrationalBasisFunction((3, 0, 0), 1, 0))

    def test_equality(self):
        self.assertEqual(self.ansatz, RovibrationalAnsatz(3, 2, 1))
        self.assertNotEqual(self.ansatz, RovibrationalAnsatz(3, 2, 0))
        self.assertNotEqual(self.ansatz, RovibrationalAnsatz(3, 1, 1))
        self.assertNotEqual(self.ansatz, RovibrationalAnsatz(2, 2, 1))

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):  # Nvibs > 0
            RovibrationalAnsatz(0, 2, 0)
        with self.assertRaises(ValueError):  # Vmax >= 0
            RovibrationalAnsatz(3, -1, 0)
        with self.assertRaises(ValueError):  # J >= 0
            RovibrationalAnsatz(3, 2, -1)
