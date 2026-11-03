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

import itertools
import math
import unittest

import numpy as np

from rovib import (
    RigidRotorHamiltonian,
    RovibrationalAnsatz,
    RovibrationalBasisFunction,
)
from rovib.matrix_elements.rigid_rotor_hamiltonian_elements import (
    rigid_rotor_matrix_element,
    rigid_rotor_non_zero_indices,
)


def _rotor_matrix(term, J):
    """The matrix of ``term`` for a single (v, J) block, ordered by K."""
    ansatz = RovibrationalAnsatz(1, 0, J)
    return np.array(
        [
            [rigid_rotor_matrix_element(bra, term, ket) for ket in ansatz]
            for bra in ansatz
        ]
    )


class TestRigidRotorMatrixElement(unittest.TestCase):
    def setUp(self):
        self.gamma = (1.0, 3.0, 2.0)
        self.term = RigidRotorHamiltonian(self.gamma)

    def test_J0(self):
        bf = RovibrationalBasisFunction((1, 2), 0, 0)
        value = rigid_rotor_matrix_element(bf, self.term, bf)
        self.assertIsInstance(value, complex)
        self.assertEqual(value, 0.0)

    def test_diagonal(self):
        bf = RovibrationalBasisFunction((1, 2), 2, 1)
        # (1 + 3) / 2 * (6 - 1) + 2 * 1
        self.assertEqual(rigid_rotor_matrix_element(bf, self.term, bf), 12.0)

    def test_delta_K_minus_2(self):
        bra = RovibrationalBasisFunction((1, 2), 2, -1)
        ket = RovibrationalBasisFunction((1, 2), 2, 1)
        # (1 - 3) / 4 * C^-_{2,1} * C^-_{2,0}
        expected = -0.5 * math.sqrt(6.0) * math.sqrt(6.0)
        self.assertAlmostEqual(
            rigid_rotor_matrix_element(bra, self.term, ket), expected
        )

    def test_delta_K_plus_2(self):
        bra = RovibrationalBasisFunction((1, 2), 2, 2)
        ket = RovibrationalBasisFunction((1, 2), 2, 0)
        # (1 - 3) / 4 * C^+_{2,0} * C^+_{2,1}
        expected = -0.5 * math.sqrt(6.0) * math.sqrt(4.0)
        self.assertAlmostEqual(
            rigid_rotor_matrix_element(bra, self.term, ket), expected
        )

    def test_hermitian(self):
        for J in range(5):
            with self.subTest(J=J):
                matrix = _rotor_matrix(self.term, J)
                np.testing.assert_allclose(matrix, matrix.conj().T)

    def test_delta_K_odd_or_large(self):
        ket = RovibrationalBasisFunction((1, 2), 3, 0)
        for K in (-3, -1, 1, 3):
            with self.subTest(K=K):
                bra = RovibrationalBasisFunction((1, 2), 3, K)
                self.assertEqual(
                    rigid_rotor_matrix_element(bra, self.term, ket), 0.0
                )

    def test_off_diagonal_in_v(self):
        bra = RovibrationalBasisFunction((1, 2), 2, 0)
        ket = RovibrationalBasisFunction((1, 1), 2, 0)
        self.assertEqual(rigid_rotor_matrix_element(bra, self.term, ket), 0.0)

    def test_off_diagonal_in_J(self):
        bra = RovibrationalBasisFunction((1, 2), 2, 0)
        ket = RovibrationalBasisFunction((1, 2), 1, 0)
        self.assertEqual(rigid_rotor_matrix_element(bra, self.term, ket), 0.0)

    def test_symmetric_top(self):
        # gamma_x = gamma_y, so the matrix is diagonal in K.
        term = RigidRotorHamiltonian((2.0, 2.0, 5.0))
        J = 3
        K = np.arange(-J, J + 1)
        expected = 2.0 * J * (J + 1) + 3.0 * K**2
        np.testing.assert_allclose(_rotor_matrix(term, J), np.diag(expected))

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


class TestRigidRotorNonZeroIndices(unittest.TestCase):
    def setUp(self):
        self.term = RigidRotorHamiltonian((1.0, 3.0, 2.0))
        self.ansatz = RovibrationalAnsatz(2, 1, 2)

    def test_J0(self):
        ansatz = RovibrationalAnsatz(2, 1, 0)
        pairs = list(rigid_rotor_non_zero_indices(ansatz, self.term, ansatz))
        self.assertEqual(pairs, [(bf, bf) for bf in ansatz])

    def test_same_space(self):
        pairs = list(
            rigid_rotor_non_zero_indices(self.ansatz, self.term, self.ansatz)
        )
        # For J = 2 each v block has 5 diagonal pairs and 3 pairs each with
        # Delta K = -2 and Delta K = +2.
        self.assertEqual(len(pairs), 4 * 11)
        self.assertEqual(len(set(pairs)), len(pairs))
        for bra, ket in pairs:
            self.assertEqual(bra.v, ket.v)
            self.assertEqual(bra.J, ket.J)
            self.assertIn(bra.K - ket.K, (-2, 0, 2))

    def test_all_non_zero_elements_generated(self):
        pairs = set(
            rigid_rotor_non_zero_indices(self.ansatz, self.term, self.ansatz)
        )
        for bra, ket in itertools.product(self.ansatz, repeat=2):
            if (bra, ket) not in pairs:
                self.assertEqual(
                    rigid_rotor_matrix_element(bra, self.term, ket), 0.0
                )

    def test_different_spaces(self):
        bra_space = RovibrationalAnsatz(2, 0, 2)
        pairs = list(
            rigid_rotor_non_zero_indices(bra_space, self.term, self.ansatz)
        )
        self.assertEqual(len(pairs), 11)
        for bra, ket in pairs:
            self.assertIn(bra, bra_space)
            self.assertIn(ket, self.ansatz)

    def test_different_J(self):
        bra_space = RovibrationalAnsatz(2, 1, 1)
        pairs = rigid_rotor_non_zero_indices(bra_space, self.term, self.ansatz)
        self.assertEqual(list(pairs), [])
