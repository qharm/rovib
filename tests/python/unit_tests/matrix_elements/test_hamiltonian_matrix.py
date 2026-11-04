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

import math
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
from rovib.matrix_elements.vibrational_matrix_elements import (
    vibrational_matrix,
)
from rovib.operators.j_operator import JOperator
from rovib.operators.normal_mode import NormalMode


def _vibrational_matrix(term, bra_ansatz, ket_ansatz):
    """The matrix of ``term`` summed from its products."""
    matrix = sum(
        vibrational_matrix(
            bra_ansatz.vibrational,
            operators,
            ket_ansatz.vibrational,
            reducer,
        )
        for operators, reducer, _ in term.terms(ket_ansatz.rotational)
    )
    return np.kron(matrix, np.eye(len(ket_ansatz.rotational)))


def _harmonic_diagonal(frequencies, ansatz):
    """The diagonal of the harmonic oscillator Hamiltonian in ``ansatz``."""
    return [np.dot(frequencies, np.add(bf.v, 0.5)) for bf in ansatz]


def _rotor_matrix(term, J):
    """The matrix of ``term`` for a single (v, J) block, ordered by K."""
    ansatz = RovibrationalAnsatz(1, 0, J)
    return build_matrix(ansatz, term, ansatz)


def _closed_form(gamma, J):
    """The elements from the rigid rotor page of the background."""
    x, y, z = gamma

    def ladder(K, sign):
        return math.sqrt(J * (J + 1) - K * (K + sign))

    matrix = np.zeros((2 * J + 1, 2 * J + 1))
    for K in range(-J, J + 1):
        k = K + J
        matrix[k, k] = 0.5 * (x + y) * (J * (J + 1) - K**2) + z * K**2
        for sign in (-1, 1):
            if abs(K + 2 * sign) <= J:
                matrix[k + 2 * sign, k] = (
                    0.25 * (x - y) * ladder(K, sign) * ladder(K + sign, sign)
                )
    return matrix


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


def _two(coefficients):
    return 2.0 * coefficients[0]


class _ProductTerm(HamiltonianTerm):
    """2 Q_0 tensor J_z, for a mode with a frequency of one."""

    def __init__(self):
        self._operator = _SingleProduct()

    @property
    def nvib(self):
        return 1

    def terms(self, kets):
        return (([self._operator], _two, [JOperator(kets.J, 2)]),)


class _SingleProduct:
    """The sum of products with one element: Q_0 with a coefficient of one."""

    def vibrational_operators(self):
        return ((NormalMode(0, 1.0),),)

    def coefficient(self, operators):
        return 1.0


class TestBuildMatrix(unittest.TestCase):
    def setUp(self):
        self.ansatz = RovibrationalAnsatz(3, 2, 1)
        self.other = RovibrationalAnsatz(1, 1, 1)

    def test_harmonic(self):
        frequencies = [1.0, 2.0, 3.0]
        term = HarmonicHamiltonian(frequencies)
        matrix = build_matrix(self.ansatz, term, self.ansatz)
        self.assertEqual(matrix.dtype, np.complex128)
        expected = np.diag(_harmonic_diagonal(frequencies, self.ansatz))
        np.testing.assert_allclose(matrix, expected)
        # The off-diagonal elements are exactly zero.
        np.testing.assert_array_equal(matrix[expected == 0.0], 0.0)

    def test_any_term_with_terms_method(self):
        # Q_0 tensor J_z for one mode, with no term-specific code.
        matrix = build_matrix(self.other, _ProductTerm(), self.other)
        q = np.sqrt(0.5) * np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(
            matrix, 2.0 * np.kron(q, np.diag([-1.0, 0.0, 1.0]))
        )

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
        expected = np.kron(
            np.eye(len(self.ansatz.vibrational)),
            _closed_form((1.0, 3.0, 2.0), self.ansatz.J),
        )
        np.testing.assert_allclose(matrix, expected, atol=1.0e-12)

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

    def test_too_many_modes(self):
        term = HarmonicHamiltonian([1.0, 2.0, 3.0])
        other = RovibrationalAnsatz(4, 1, 1)
        with self.assertRaises(ValueError):
            build_matrix(other, term, other)

    def test_mode_mismatch_all_coefficients_zero(self):
        term = AnharmonicPotential(
            [1.0, 2.0, 3.0], np.zeros((3, 3, 3)), np.zeros((3, 3, 3, 3))
        )
        other = RovibrationalAnsatz(2, 2, 1)
        with self.assertRaises(ValueError):
            build_matrix(other, term, other)

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

        expected = np.kron(
            np.eye(len(self.ansatz.vibrational)),
            _closed_form((1.0, 3.0, 2.0), self.ansatz.J),
        )
        np.testing.assert_allclose(matrix, expected, atol=1.0e-12)

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


class TestRigidRotorMatrix(unittest.TestCase):
    def setUp(self):
        self.gamma = (1.0, 3.0, 2.0)
        self.term = RigidRotorHamiltonian(self.gamma)

    def test_J0(self):
        matrix = _rotor_matrix(self.term, 0)
        self.assertEqual(matrix.dtype, np.complex128)
        np.testing.assert_array_equal(matrix, [[0.0]])

    def test_diagonal(self):
        # (1 + 3) / 2 * (6 - 1) + 2 * 1
        self.assertAlmostEqual(_rotor_matrix(self.term, 2)[3, 3], 12.0)

    def test_delta_K_minus_2(self):
        # (1 - 3) / 4 * C^-_{2,1} * C^-_{2,0}, with K = 1 to K' = -1.
        expected = -0.5 * math.sqrt(6.0) * math.sqrt(6.0)
        self.assertAlmostEqual(_rotor_matrix(self.term, 2)[1, 3], expected)

    def test_delta_K_plus_2(self):
        # (1 - 3) / 4 * C^+_{2,0} * C^+_{2,1}, with K = 0 to K' = 2.
        expected = -0.5 * math.sqrt(6.0) * math.sqrt(4.0)
        self.assertAlmostEqual(_rotor_matrix(self.term, 2)[4, 2], expected)

    def test_matches_closed_form(self):
        for J in range(7):
            with self.subTest(J=J):
                np.testing.assert_allclose(
                    _rotor_matrix(self.term, J),
                    _closed_form(self.gamma, J),
                    atol=1.0e-12,
                )

    def test_hermitian_and_real(self):
        for J in range(5):
            with self.subTest(J=J):
                matrix = _rotor_matrix(self.term, J)
                np.testing.assert_allclose(matrix, matrix.conj().T)
                np.testing.assert_array_equal(matrix.imag, 0.0)

    def test_delta_K_selection_rule(self):
        matrix = _rotor_matrix(self.term, 3)
        for b, k in np.ndindex(matrix.shape):
            if b - k not in (-2, 0, 2):
                self.assertEqual(matrix[b, k], 0.0)

    def test_diagonal_in_v(self):
        ansatz = RovibrationalAnsatz(2, 2, 2)
        matrix = build_matrix(ansatz, self.term, ansatz)
        self.assertGreater(np.count_nonzero(matrix), 0)
        for b, k in zip(*np.nonzero(matrix)):
            bra = ansatz.basis_function(b)
            ket = ansatz.basis_function(k)
            self.assertEqual(bra.v, ket.v)

    def test_same_block_for_each_v(self):
        ansatz = RovibrationalAnsatz(2, 1, 1)
        matrix = build_matrix(ansatz, self.term, ansatz)
        block = _rotor_matrix(self.term, 1)
        np.testing.assert_allclose(matrix, np.kron(np.eye(4), block))

    def test_different_vmax(self):
        # The v = (0, 1) function has a different index in each space.
        bra_ansatz = RovibrationalAnsatz(2, 1, 2)
        ket_ansatz = RovibrationalAnsatz(2, 2, 2)
        block = build_matrix(bra_ansatz, self.term, ket_ansatz)
        self.assertEqual(block.shape, (len(bra_ansatz), len(ket_ansatz)))
        full = build_matrix(ket_ansatz, self.term, ket_ansatz)
        rows = [ket_ansatz.index(bra) for bra in bra_ansatz]
        np.testing.assert_allclose(block, full[rows, :])

    def test_different_J(self):
        bra_ansatz = RovibrationalAnsatz(2, 1, 1)
        ket_ansatz = RovibrationalAnsatz(2, 1, 2)
        block = build_matrix(bra_ansatz, self.term, ket_ansatz)
        self.assertEqual(block.shape, (len(bra_ansatz), len(ket_ansatz)))
        np.testing.assert_array_equal(block, 0.0)

    def test_different_number_of_modes(self):
        bra_ansatz = RovibrationalAnsatz(1, 1, 1)
        ket_ansatz = RovibrationalAnsatz(2, 1, 1)
        block = build_matrix(bra_ansatz, self.term, ket_ansatz)
        np.testing.assert_array_equal(block, 0.0)

    def test_element_order_matches_ansatz(self):
        ansatz = RovibrationalAnsatz(1, 1, 1)
        matrix = build_matrix(ansatz, self.term, ansatz)
        bf = RovibrationalBasisFunction((1,), 1, 0)
        i = ansatz.index(bf)
        # (1 + 3) / 2 * 2
        self.assertAlmostEqual(matrix[i, i], 4.0)

    def test_symmetric_top(self):
        # gamma_x = gamma_y, so the matrix is diagonal in K.
        term = RigidRotorHamiltonian((2.0, 2.0, 5.0))
        J = 3
        K = np.arange(-J, J + 1)
        expected = 2.0 * J * (J + 1) + 3.0 * K**2
        np.testing.assert_allclose(
            _rotor_matrix(term, J), np.diag(expected), atol=1.0e-12
        )

    def test_J1_levels(self):
        x, y, z = self.gamma
        energies = np.linalg.eigvalsh(_rotor_matrix(self.term, 1))
        np.testing.assert_allclose(energies, sorted([x + y, x + z, y + z]))

    def test_J2_levels(self):
        x, y, z = self.gamma
        s = x + y + z
        d = math.sqrt(s**2 - 3.0 * (x * y + y * z + z * x))
        expected = [s + 3.0 * x, s + 3.0 * y, s + 3.0 * z]
        expected += [2.0 * s - 2.0 * d, 2.0 * s + 2.0 * d]
        energies = np.linalg.eigvalsh(_rotor_matrix(self.term, 2))
        np.testing.assert_allclose(energies, sorted(expected))
