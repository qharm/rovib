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
    RotationalBasisFunction,
    VibrationalBasisFunction,
)


class TestRotationalAnsatz(unittest.TestCase):
    def setUp(self):
        self.ansatz = RotationalAnsatz(2)

    def test_state(self):
        self.assertEqual(self.ansatz.J, 2)

    def test_len(self):
        self.assertEqual(len(self.ansatz), 5)
        self.assertEqual(len(RotationalAnsatz(0)), 1)

    def test_iteration_order(self):
        expected = [RotationalBasisFunction(1, K) for K in (-1, 0, 1)]
        self.assertEqual(list(RotationalAnsatz(1)), expected)

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
        self.assertIn(RotationalBasisFunction(2, -2), self.ansatz)
        # Wrong J
        self.assertNotIn(RotationalBasisFunction(1, 0), self.ansatz)
        self.assertNotIn(VibrationalBasisFunction((0,)), self.ansatz)

    def test_index_not_in_ansatz(self):
        with self.assertRaises(ValueError):
            self.ansatz.index(RotationalBasisFunction(1, 0))

    def test_equality(self):
        self.assertEqual(self.ansatz, RotationalAnsatz(2))
        self.assertNotEqual(self.ansatz, RotationalAnsatz(1))

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            RotationalAnsatz(-1)
