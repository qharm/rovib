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

from rovib import AnharmonicPotential, HamiltonianTerm


class TestAnharmonicPotential(unittest.TestCase):
    def setUp(self):
        self.frequencies = np.array([1.0, 2.0])
        self.cubic = np.zeros((2, 2, 2))
        self.cubic[0, 0, 1] = self.cubic[0, 1, 0] = self.cubic[1, 0, 0] = 3.0
        self.quartic = np.full((2, 2, 2, 2), 4.0)
        self.term = AnharmonicPotential(
            self.frequencies, self.cubic, self.quartic
        )

    def test_is_hamiltonian_term(self):
        self.assertIsInstance(self.term, HamiltonianTerm)

    def test_state(self):
        np.testing.assert_array_equal(self.term.frequencies, self.frequencies)
        np.testing.assert_array_equal(self.term.cubic, self.cubic)
        np.testing.assert_array_equal(self.term.quartic, self.quartic)
        self.assertEqual(self.term.nvib, 2)

    def test_state_is_read_only(self):
        with self.assertRaises(ValueError):
            self.term.frequencies[0] = 1.0
        with self.assertRaises(ValueError):
            self.term.cubic[0, 0, 0] = 1.0
        with self.assertRaises(ValueError):
            self.term.quartic[0, 0, 0, 0] = 1.0

    def test_input_is_copied(self):
        self.frequencies[0] = 100.0
        self.cubic[0, 0, 1] = 100.0
        self.quartic[0, 0, 0, 0] = 100.0
        self.assertEqual(self.term.frequencies[0], 1.0)
        self.assertEqual(self.term.cubic[0, 0, 1], 3.0)
        self.assertEqual(self.term.quartic[0, 0, 0, 0], 4.0)

    def test_bad_frequencies_shape(self):
        with self.assertRaises(ValueError):
            AnharmonicPotential(np.ones((2, 1)), self.cubic, self.quartic)
        with self.assertRaises(ValueError):
            AnharmonicPotential([], np.zeros((0,) * 3), np.zeros((0,) * 4))

    def test_non_positive_frequency(self):
        with self.assertRaises(ValueError):
            AnharmonicPotential([1.0, 0.0], self.cubic, self.quartic)
        with self.assertRaises(ValueError):
            AnharmonicPotential([1.0, -2.0], self.cubic, self.quartic)

    def test_bad_cubic_shape(self):
        for shape in ((2, 2), (2, 2, 3), (3, 3, 3), (2, 2, 2, 2)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    AnharmonicPotential(
                        self.frequencies, np.zeros(shape), self.quartic
                    )

    def test_bad_quartic_shape(self):
        for shape in ((2, 2, 2), (2, 2, 2, 3), (3, 3, 3, 3)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    AnharmonicPotential(
                        self.frequencies, self.cubic, np.zeros(shape)
                    )

    def test_asymmetric_cubic(self):
        cubic = np.zeros((2, 2, 2))
        cubic[0, 0, 1] = 1.0
        with self.assertRaises(ValueError):
            AnharmonicPotential(self.frequencies, cubic, self.quartic)

    def test_asymmetric_quartic(self):
        quartic = np.zeros((2, 2, 2, 2))
        quartic[0, 1, 1, 1] = 1.0
        with self.assertRaises(ValueError):
            AnharmonicPotential(self.frequencies, self.cubic, quartic)
