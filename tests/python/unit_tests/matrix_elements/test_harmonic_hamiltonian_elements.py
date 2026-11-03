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
    HarmonicHamiltonian,
    RovibrationalAnsatz,
    RovibrationalBasisFunction,
)
from rovib.matrix_elements.harmonic_hamiltonian_elements import (
    harmonic_matrix_element,
    harmonic_non_zero_indices,
)


class TestHarmonicMatrixElement(unittest.TestCase):
    def setUp(self):
        self.term = HarmonicHamiltonian([1.0, 2.0, 3.0])

    def test_diagonal(self):
        bf = RovibrationalBasisFunction((0, 1, 2), 1, 0)
        value = harmonic_matrix_element(bf, self.term, bf)
        self.assertIsInstance(value, complex)
        self.assertEqual(value, 0.5 + 2.0 * 1.5 + 3.0 * 2.5)

    def test_ground_state(self):
        bf = RovibrationalBasisFunction((0, 0, 0), 0, 0)
        self.assertEqual(harmonic_matrix_element(bf, self.term, bf), 3.0)

    def test_off_diagonal_in_v(self):
        bra = RovibrationalBasisFunction((0, 1, 2), 1, 0)
        ket = RovibrationalBasisFunction((0, 2, 2), 1, 0)
        self.assertEqual(harmonic_matrix_element(bra, self.term, ket), 0.0)

    def test_off_diagonal_in_K(self):
        bra = RovibrationalBasisFunction((0, 1, 2), 1, 0)
        ket = RovibrationalBasisFunction((0, 1, 2), 1, 1)
        self.assertEqual(harmonic_matrix_element(bra, self.term, ket), 0.0)


class TestHarmonicNonZeroIndices(unittest.TestCase):
    def setUp(self):
        self.term = HarmonicHamiltonian([1.0, 2.0, 3.0])
        self.ansatz = RovibrationalAnsatz(3, 2, 1)

    def test_same_space(self):
        pairs = list(
            harmonic_non_zero_indices(self.ansatz, self.term, self.ansatz)
        )
        self.assertEqual(pairs, [(bf, bf) for bf in self.ansatz])

    def test_different_spaces(self):
        bra_space = RovibrationalAnsatz(3, 1, 1)
        pairs = list(
            harmonic_non_zero_indices(bra_space, self.term, self.ansatz)
        )
        self.assertEqual(len(pairs), len(bra_space))
        for bra, ket in pairs:
            self.assertEqual(bra, ket)
            self.assertIn(bra, bra_space)
            self.assertIn(ket, self.ansatz)

    def test_different_J(self):
        bra_space = RovibrationalAnsatz(3, 2, 0)
        pairs = harmonic_non_zero_indices(bra_space, self.term, self.ansatz)
        self.assertEqual(list(pairs), [])

    def test_mode_mismatch(self):
        other = RovibrationalAnsatz(2, 2, 1)
        with self.assertRaises(ValueError):
            list(harmonic_non_zero_indices(other, self.term, self.ansatz))
        with self.assertRaises(ValueError):
            list(harmonic_non_zero_indices(self.ansatz, self.term, other))
