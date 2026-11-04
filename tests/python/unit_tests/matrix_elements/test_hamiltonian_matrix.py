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
    VibrationalCoriolisHamiltonian,
    build_hamiltonian_matrix,
)
from rovib.matrix_elements.hamiltonian_matrix import build_matrix
from rovib.matrix_elements.rigid_rotor_hamiltonian_elements import (
    rigid_rotor_matrix,
)
from rovib.matrix_elements.vibrational_matrix_elements import (
    vibrational_matrix,
)


def _vibrational_matrix(term, bra_ansatz, ket_ansatz):
    """The matrix of ``term`` summed from its products."""
    matrix = sum(
        vibrational_matrix(
            bra_ansatz.vibrational,
            operators,
            ket_ansatz.vibrational,
            reducer,
        )
        for operators, reducer in term.terms()
    )
    return np.kron(matrix, np.eye(len(ket_ansatz.rotational)))


def _harmonic_diagonal(frequencies, ansatz):
    """The diagonal of the harmonic oscillator Hamiltonian in ``ansatz``."""
    return [np.dot(frequencies, np.add(bf.v, 0.5)) for bf in ansatz]


def _anharmonic_potential():
    """An AnharmonicPotential for three modes with symmetric constants."""
    cubic = np.zeros((3, 3, 3))
    cubic[0, 0, 1] = cubic[0, 1, 0] = cubic[1, 0, 0] = 0.5
    cubic[2, 2, 2] = -0.25
    quartic = np.zeros((3, 3, 3, 3))
    quartic[1, 1, 1, 1] = 0.75
    return AnharmonicPotential([1.0, 2.0, 3.0], cubic, quartic)


def _vibrational_coriolis_hamiltonian():
    """A VibrationalCoriolisHamiltonian for three modes."""
    zeta = np.zeros((3, 3, 3))
    zeta[0, 0, 2] = zeta[1, 2, 2] = 0.5
    zeta[2, 0, 0] = zeta[2, 2, 1] = -0.5
    a = np.zeros((3, 3, 3))
    a[0] = np.diag([0.1, 0.2, 0.3])
    a[1, 0, 1] = a[1, 1, 0] = 0.4
    return VibrationalCoriolisHamiltonian(
        [1.0, 2.0, 3.0], zeta, a, [0.25, 1.0, 0.5]
    )


class TestBuildMatrix(unittest.TestCase):
    def setUp(self):
        self.ansatz = RovibrationalAnsatz(3, 2, 1)

    def test_harmonic(self):
        frequencies = [1.0, 2.0, 3.0]
        term = HarmonicHamiltonian(frequencies)
        matrix = build_matrix(self.ansatz, term, self.ansatz)
        self.assertEqual(matrix.dtype, np.complex128)
        expected = np.diag(_harmonic_diagonal(frequencies, self.ansatz))
        np.testing.assert_allclose(matrix, expected)
        # The off-diagonal elements are exactly zero.
        np.testing.assert_array_equal(matrix[expected == 0.0], 0.0)

    def test_harmonic_element(self):
        term = HarmonicHamiltonian([1.0, 2.0, 3.0])
        matrix = build_matrix(self.ansatz, term, self.ansatz)
        i = self.ansatz.index(RovibrationalBasisFunction((0, 1, 2), 1, 0))
        self.assertAlmostEqual(matrix[i, i], 0.5 + 2.0 * 1.5 + 3.0 * 2.5)

    def test_harmonic_ground_state(self):
        term = HarmonicHamiltonian([1.0, 2.0, 3.0])
        ansatz = RovibrationalAnsatz(3, 0, 0)
        matrix = build_matrix(ansatz, term, ansatz)
        np.testing.assert_allclose(matrix, [[3.0]])

    def test_harmonic_block(self):
        term = HarmonicHamiltonian([1.0, 2.0, 3.0])
        bra_space = RovibrationalAnsatz(3, 1, 1)
        block = build_matrix(bra_space, term, self.ansatz)
        full = build_matrix(self.ansatz, term, self.ansatz)
        rows = [self.ansatz.index(bra) for bra in bra_space]
        np.testing.assert_array_equal(block, full[rows, :])

    def test_harmonic_different_J(self):
        term = HarmonicHamiltonian([1.0, 2.0, 3.0])
        bra_space = RovibrationalAnsatz(3, 2, 0)
        block = build_matrix(bra_space, term, self.ansatz)
        self.assertEqual(block.shape, (len(bra_space), len(self.ansatz)))
        np.testing.assert_array_equal(block, 0.0)

    def test_harmonic_mode_mismatch(self):
        term = HarmonicHamiltonian([1.0, 2.0, 3.0])
        other = RovibrationalAnsatz(2, 2, 1)
        with self.assertRaises(ValueError):
            build_matrix(other, term, self.ansatz)
        with self.assertRaises(ValueError):
            build_matrix(self.ansatz, term, other)

    def test_anharmonic(self):
        term = _anharmonic_potential()
        matrix = build_matrix(self.ansatz, term, self.ansatz)
        self.assertEqual(matrix.dtype, np.complex128)
        expected = _vibrational_matrix(term, self.ansatz, self.ansatz)
        np.testing.assert_allclose(matrix, expected, atol=1.0e-12)

    def test_vibrational_coriolis(self):
        term = _vibrational_coriolis_hamiltonian()
        matrix = build_matrix(self.ansatz, term, self.ansatz)
        self.assertEqual(matrix.dtype, np.complex128)
        expected = _vibrational_matrix(term, self.ansatz, self.ansatz)
        np.testing.assert_allclose(matrix, expected, atol=1.0e-12)

    def test_rigid_rotor(self):
        term = RigidRotorHamiltonian([1.0, 3.0, 2.0])
        matrix = build_matrix(self.ansatz, term, self.ansatz)
        self.assertEqual(matrix.dtype, np.complex128)
        expected = rigid_rotor_matrix(self.ansatz, term, self.ansatz)
        np.testing.assert_array_equal(matrix, expected)

    def test_vibrational_term_diagonal_in_K(self):
        term = _anharmonic_potential()
        matrix = build_matrix(self.ansatz, term, self.ansatz)
        self.assertGreater(np.count_nonzero(matrix), 0)
        for b, k in zip(*np.nonzero(matrix)):
            bra = self.ansatz.basis_function(b)
            ket = self.ansatz.basis_function(k)
            self.assertEqual(bra.K, ket.K)

    def test_vibrational_term_different_J(self):
        term = _anharmonic_potential()
        bra_space = RovibrationalAnsatz(3, 2, 0)
        block = build_matrix(bra_space, term, self.ansatz)
        self.assertEqual(block.shape, (len(bra_space), len(self.ansatz)))
        np.testing.assert_array_equal(block, 0.0)

    def test_vibrational_coriolis_mode_mismatch(self):
        term = _vibrational_coriolis_hamiltonian()
        other = RovibrationalAnsatz(2, 2, 1)
        with self.assertRaises(ValueError):
            build_matrix(other, term, self.ansatz)

    def test_block(self):
        term = _anharmonic_potential()
        bra_space = RovibrationalAnsatz(3, 1, 1)
        block = build_matrix(bra_space, term, self.ansatz)
        self.assertEqual(block.shape, (len(bra_space), len(self.ansatz)))
        np.testing.assert_allclose(
            block, _vibrational_matrix(term, bra_space, self.ansatz)
        )

    def test_mode_mismatch(self):
        term = _anharmonic_potential()
        other = RovibrationalAnsatz(2, 2, 1)
        with self.assertRaises(ValueError):
            build_matrix(other, term, self.ansatz)
        with self.assertRaises(ValueError):
            build_matrix(self.ansatz, term, other)

    def test_unknown_term(self):
        with self.assertRaises(TypeError):
            build_matrix(self.ansatz, HamiltonianTerm(), self.ansatz)


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

        expected = _harmonic_diagonal(self.term.frequencies, self.ansatz)
        np.testing.assert_allclose(np.diag(matrix), expected)
        np.testing.assert_array_equal(matrix - np.diag(np.diag(matrix)), 0.0)

    def test_rigid_rotor(self):
        term = RigidRotorHamiltonian([1.0, 3.0, 2.0])
        matrix = build_hamiltonian_matrix(Hamiltonian((term,)), self.ansatz)
        np.testing.assert_allclose(matrix, matrix.conj().T)

        expected = rigid_rotor_matrix(self.ansatz, term, self.ansatz)
        np.testing.assert_array_equal(matrix, expected)

    def test_anharmonic(self):
        term = _anharmonic_potential()
        matrix = build_hamiltonian_matrix(Hamiltonian((term,)), self.ansatz)
        np.testing.assert_allclose(matrix, matrix.conj().T)

        expected = _vibrational_matrix(term, self.ansatz, self.ansatz)
        np.testing.assert_allclose(matrix, expected, atol=1.0e-12)

    def test_vibrational_coriolis(self):
        term = _vibrational_coriolis_hamiltonian()
        matrix = build_hamiltonian_matrix(Hamiltonian((term,)), self.ansatz)
        np.testing.assert_allclose(matrix, matrix.conj().T)

        expected = _vibrational_matrix(term, self.ansatz, self.ansatz)
        np.testing.assert_allclose(matrix, expected, atol=1.0e-12)

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
