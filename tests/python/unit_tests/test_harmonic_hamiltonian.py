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

from rovib import HamiltonianTerm, HarmonicHamiltonian, RotationalAnsatz


class TestHarmonicHamiltonian(unittest.TestCase):
    def setUp(self):
        self.frequencies = np.array([1.0, 2.0, 3.0])
        self.term = HarmonicHamiltonian(self.frequencies)

    def test_is_hamiltonian_term(self):
        self.assertIsInstance(self.term, HamiltonianTerm)

    def test_state(self):
        np.testing.assert_array_equal(self.term.frequencies, self.frequencies)
        self.assertEqual(self.term.nvib, 3)

    def test_terms(self):
        ((operators, reducer, rotational),) = self.term.terms(
            RotationalAnsatz(1)
        )
        self.assertEqual(rotational, [])
        self.assertEqual(len(operators), 1)
        self.assertEqual(reducer([2.0]), 2.0)
        self.assertEqual(operators[0].coefficient(()), 3.0)

    def test_frequencies_are_read_only(self):
        with self.assertRaises(ValueError):
            self.term.frequencies[0] = 1.0

    def test_input_is_copied(self):
        self.frequencies[0] = 100.0
        self.assertEqual(self.term.frequencies[0], 1.0)

    def test_bad_frequencies_shape(self):
        with self.assertRaises(ValueError):
            HarmonicHamiltonian(np.ones((3, 1)))
        with self.assertRaises(ValueError):
            HarmonicHamiltonian([])

    def test_non_positive_frequency(self):
        with self.assertRaises(ValueError):
            HarmonicHamiltonian([1.0, 0.0, 3.0])
        with self.assertRaises(ValueError):
            HarmonicHamiltonian([1.0, -2.0, 3.0])
