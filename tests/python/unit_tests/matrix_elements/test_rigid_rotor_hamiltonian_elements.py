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
    RigidRotorHamiltonian,
    RovibrationalAnsatz,
    RovibrationalBasisFunction,
)
from rovib.matrix_elements.rigid_rotor_hamiltonian_elements import (
    rigid_rotor_matrix,
)


def _rotor_matrix(term, J):
    """The matrix of ``term`` for a single (v, J) block, ordered by K."""
    ansatz = RovibrationalAnsatz(1, 0, J)
    return rigid_rotor_matrix(ansatz, term, ansatz)


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
        matrix = rigid_rotor_matrix(ansatz, self.term, ansatz)
        self.assertGreater(np.count_nonzero(matrix), 0)
        for b, k in zip(*np.nonzero(matrix)):
            bra = ansatz.basis_function(b)
            ket = ansatz.basis_function(k)
            self.assertEqual(bra.v, ket.v)

    def test_same_block_for_each_v(self):
        ansatz = RovibrationalAnsatz(2, 1, 1)
        matrix = rigid_rotor_matrix(ansatz, self.term, ansatz)
        block = _rotor_matrix(self.term, 1)
        np.testing.assert_allclose(matrix, np.kron(np.eye(4), block))

    def test_different_vmax(self):
        # The v = (0, 1) function has a different index in each space.
        bra_ansatz = RovibrationalAnsatz(2, 1, 2)
        ket_ansatz = RovibrationalAnsatz(2, 2, 2)
        block = rigid_rotor_matrix(bra_ansatz, self.term, ket_ansatz)
        self.assertEqual(block.shape, (len(bra_ansatz), len(ket_ansatz)))
        full = rigid_rotor_matrix(ket_ansatz, self.term, ket_ansatz)
        rows = [ket_ansatz.index(bra) for bra in bra_ansatz]
        np.testing.assert_allclose(block, full[rows, :])

    def test_different_J(self):
        bra_ansatz = RovibrationalAnsatz(2, 1, 1)
        ket_ansatz = RovibrationalAnsatz(2, 1, 2)
        block = rigid_rotor_matrix(bra_ansatz, self.term, ket_ansatz)
        self.assertEqual(block.shape, (len(bra_ansatz), len(ket_ansatz)))
        np.testing.assert_array_equal(block, 0.0)

    def test_different_number_of_modes(self):
        bra_ansatz = RovibrationalAnsatz(1, 1, 1)
        ket_ansatz = RovibrationalAnsatz(2, 1, 1)
        block = rigid_rotor_matrix(bra_ansatz, self.term, ket_ansatz)
        np.testing.assert_array_equal(block, 0.0)

    def test_element_order_matches_ansatz(self):
        ansatz = RovibrationalAnsatz(1, 1, 1)
        matrix = rigid_rotor_matrix(ansatz, self.term, ansatz)
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
