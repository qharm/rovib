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

from rovib import (
    AnharmonicPotential,
    Hamiltonian,
    HamiltonianTerm,
    HarmonicHamiltonian,
    RigidRotorHamiltonian,
    RovibrationalAnsatz,
    RovibrationalBasisFunction,
    build_hamiltonian_matrix,
    matrix_element,
    non_zero_indices,
)
from rovib.anharmonic_potential_elements import (
    anharmonic_matrix_element,
    anharmonic_non_zero_indices,
)
from rovib.harmonic_hamiltonian_elements import (
    harmonic_matrix_element,
    harmonic_non_zero_indices,
)
from rovib.rigid_rotor_hamiltonian_elements import (
    rigid_rotor_matrix_element,
    rigid_rotor_non_zero_indices,
)


def _anharmonic_potential():
    """An AnharmonicPotential for three modes with symmetric constants."""
    cubic = np.zeros((3, 3, 3))
    cubic[0, 0, 1] = cubic[0, 1, 0] = cubic[1, 0, 0] = 0.5
    cubic[2, 2, 2] = -0.25
    quartic = np.zeros((3, 3, 3, 3))
    quartic[1, 1, 1, 1] = 0.75
    return AnharmonicPotential([1.0, 2.0, 3.0], cubic, quartic)


class TestNonZeroIndices(unittest.TestCase):
    def test_harmonic(self):
        term = HarmonicHamiltonian(np.ones(3))
        ansatz = RovibrationalAnsatz(3, 1, 0)
        self.assertEqual(
            list(non_zero_indices(ansatz, term, ansatz)),
            list(harmonic_non_zero_indices(ansatz, term, ansatz)),
        )

    def test_rigid_rotor(self):
        term = RigidRotorHamiltonian([1.0, 3.0, 2.0])
        ansatz = RovibrationalAnsatz(3, 1, 2)
        self.assertEqual(
            list(non_zero_indices(ansatz, term, ansatz)),
            list(rigid_rotor_non_zero_indices(ansatz, term, ansatz)),
        )

    def test_anharmonic(self):
        term = _anharmonic_potential()
        ansatz = RovibrationalAnsatz(3, 2, 1)
        self.assertEqual(
            list(non_zero_indices(ansatz, term, ansatz)),
            list(anharmonic_non_zero_indices(ansatz, term, ansatz)),
        )

    def test_unknown_term(self):
        ansatz = RovibrationalAnsatz(3, 1, 0)
        with self.assertRaises(TypeError):
            non_zero_indices(ansatz, HamiltonianTerm(), ansatz)


class TestMatrixElement(unittest.TestCase):
    def setUp(self):
        self.bf = RovibrationalBasisFunction((1, 0, 2), 0, 0)

    def test_harmonic(self):
        term = HarmonicHamiltonian([1.0, 2.0, 3.0])
        self.assertEqual(
            matrix_element(self.bf, term, self.bf),
            harmonic_matrix_element(self.bf, term, self.bf),
        )

    def test_rigid_rotor(self):
        term = RigidRotorHamiltonian([1.0, 3.0, 2.0])
        bra = RovibrationalBasisFunction((1, 0, 2), 2, 2)
        ket = RovibrationalBasisFunction((1, 0, 2), 2, 0)
        self.assertEqual(
            matrix_element(bra, term, ket),
            rigid_rotor_matrix_element(bra, term, ket),
        )

    def test_anharmonic(self):
        term = _anharmonic_potential()
        bra = RovibrationalBasisFunction((2, 1, 2), 0, 0)
        self.assertEqual(
            matrix_element(bra, term, self.bf),
            anharmonic_matrix_element(bra, term, self.bf),
        )

    def test_unknown_term(self):
        with self.assertRaises(TypeError):
            matrix_element(self.bf, HamiltonianTerm(), self.bf)


class TestBuildHamiltonianMatrix(unittest.TestCase):
    def setUp(self):
        self.term = HarmonicHamiltonian([1.0, 2.0, 3.0])
        self.ansatz = RovibrationalAnsatz(3, 2, 1)

    def test_harmonic(self):
        matrix = build_hamiltonian_matrix(
            Hamiltonian((self.term,)), self.ansatz
        )
        size = len(self.ansatz)
        self.assertEqual(matrix.shape, (size, size))
        self.assertEqual(matrix.dtype, np.complex128)
        np.testing.assert_array_equal(matrix, matrix.conj().T)

        expected = [
            harmonic_matrix_element(bf, self.term, bf) for bf in self.ansatz
        ]
        np.testing.assert_array_equal(np.diag(matrix), expected)
        np.testing.assert_array_equal(matrix - np.diag(np.diag(matrix)), 0.0)

    def test_rigid_rotor(self):
        term = RigidRotorHamiltonian([1.0, 3.0, 2.0])
        matrix = build_hamiltonian_matrix(Hamiltonian((term,)), self.ansatz)
        np.testing.assert_array_equal(matrix, matrix.conj().T)

        expected = np.zeros_like(matrix)
        for bra, ket in rigid_rotor_non_zero_indices(
            self.ansatz, term, self.ansatz
        ):
            b = self.ansatz.index(bra)
            k = self.ansatz.index(ket)
            expected[b, k] = rigid_rotor_matrix_element(bra, term, ket)
        np.testing.assert_array_equal(matrix, expected)

    def test_anharmonic(self):
        term = _anharmonic_potential()
        matrix = build_hamiltonian_matrix(Hamiltonian((term,)), self.ansatz)
        np.testing.assert_allclose(matrix, matrix.conj().T)

        expected = np.zeros_like(matrix)
        for bra, ket in anharmonic_non_zero_indices(
            self.ansatz, term, self.ansatz
        ):
            b = self.ansatz.index(bra)
            k = self.ansatz.index(ket)
            expected[b, k] = anharmonic_matrix_element(bra, term, ket)
        np.testing.assert_array_equal(matrix, expected)

    def test_harmonic_plus_rigid_rotor(self):
        rr = RigidRotorHamiltonian([1.0, 3.0, 2.0])
        total = build_hamiltonian_matrix(self.term + rr, self.ansatz)
        harmonic = build_hamiltonian_matrix(
            Hamiltonian((self.term,)), self.ansatz
        )
        rigid_rotor = build_hamiltonian_matrix(Hamiltonian((rr,)), self.ansatz)
        np.testing.assert_allclose(total, harmonic + rigid_rotor)

    def test_terms_are_summed(self):
        once = build_hamiltonian_matrix(Hamiltonian((self.term,)), self.ansatz)
        twice = build_hamiltonian_matrix(self.term + self.term, self.ansatz)
        np.testing.assert_allclose(twice, 2.0 * once)

    def test_empty_hamiltonian(self):
        matrix = build_hamiltonian_matrix(Hamiltonian(), self.ansatz)
        np.testing.assert_array_equal(matrix, 0.0)

    def test_unknown_term(self):
        with self.assertRaises(TypeError):
            build_hamiltonian_matrix(
                Hamiltonian((HamiltonianTerm(),)), self.ansatz
            )
