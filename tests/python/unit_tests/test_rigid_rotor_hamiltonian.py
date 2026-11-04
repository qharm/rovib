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

from rovib import HamiltonianTerm, RigidRotorHamiltonian, RotationalAnsatz
from rovib.operators.j_operator import JOperator


class TestRigidRotorHamiltonian(unittest.TestCase):
    def setUp(self):
        self.constants = np.array([1.0, 3.0, 2.0])
        self.term = RigidRotorHamiltonian(self.constants)

    def test_is_hamiltonian_term(self):
        self.assertIsInstance(self.term, HamiltonianTerm)

    def test_state(self):
        np.testing.assert_array_equal(
            self.term.rotational_constants, self.constants
        )

    def test_terms(self):
        terms = self.term.terms(RotationalAnsatz(2))
        self.assertEqual(len(terms), 3)
        for a, (vibrational, reducer, rotational) in enumerate(terms):
            with self.subTest(a=a):
                self.assertEqual(vibrational, [])
                self.assertEqual(reducer([]), self.constants[a])
                self.assertEqual(rotational, [JOperator(2, a)] * 2)

    def test_reducers_are_reused(self):
        # vibrational_matrix caches on the reducer.
        once = self.term.terms(RotationalAnsatz(1))
        twice = self.term.terms(RotationalAnsatz(2))
        for first, second in zip(once, twice):
            self.assertIs(first[1], second[1])

    def test_nvib(self):
        self.assertIsNone(self.term.nvib)

    def test_rotational_constants_are_read_only(self):
        with self.assertRaises(ValueError):
            self.term.rotational_constants[0] = 1.0

    def test_input_is_copied(self):
        self.constants[0] = 100.0
        self.assertEqual(self.term.rotational_constants[0], 1.0)

    def test_bad_shape(self):
        with self.assertRaises(ValueError):
            RigidRotorHamiltonian([1.0, 2.0])
        with self.assertRaises(ValueError):
            RigidRotorHamiltonian(np.ones((3, 1)))

    def test_non_positive_constant(self):
        with self.assertRaises(ValueError):
            RigidRotorHamiltonian([1.0, 0.0, 3.0])
        with self.assertRaises(ValueError):
            RigidRotorHamiltonian([1.0, -2.0, 3.0])
