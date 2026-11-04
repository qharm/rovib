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

from rovib import RotationalBasisFunction, VibrationalAnsatz
from rovib import VibrationalBasisFunction as Vib


class TestVibrationalAnsatz(unittest.TestCase):
    def setUp(self):
        self.ansatz = VibrationalAnsatz(3, 2)

    def test_state(self):
        self.assertEqual(self.ansatz.nvib, 3)
        self.assertEqual(self.ansatz.vmax, 2)

    def test_len(self):
        self.assertEqual(len(self.ansatz), 3**3)
        self.assertEqual(len(VibrationalAnsatz(3, 0)), 1)

    def test_iteration_order(self):
        expected = [Vib(v) for v in [(0, 0), (0, 1), (1, 0), (1, 1)]]
        self.assertEqual(list(VibrationalAnsatz(2, 1)), expected)

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
        self.assertIn(Vib((2, 0, 2)), self.ansatz)
        # vmax exceeded
        self.assertNotIn(Vib((3, 0, 0)), self.ansatz)
        # Wrong number of modes
        self.assertNotIn(Vib((0, 0)), self.ansatz)
        self.assertNotIn(RotationalBasisFunction(0, 0), self.ansatz)

    def test_index_not_in_ansatz(self):
        with self.assertRaises(ValueError):
            self.ansatz.index(Vib((3, 0, 0)))

    def test_equality(self):
        self.assertEqual(self.ansatz, VibrationalAnsatz(3, 2))
        self.assertNotEqual(self.ansatz, VibrationalAnsatz(3, 1))
        self.assertNotEqual(self.ansatz, VibrationalAnsatz(2, 2))

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            VibrationalAnsatz(0, 2)
        with self.assertRaises(ValueError):
            VibrationalAnsatz(3, -1)
