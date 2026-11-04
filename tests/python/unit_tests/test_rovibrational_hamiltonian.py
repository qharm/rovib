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
    Hamiltonian,
    HamiltonianTerm,
    RotationalAnsatz,
    RovibrationalAnsatz,
    RovibrationalHamiltonian,
    build_hamiltonian_matrix,
)
from rovib.matrix_elements.hamiltonian_matrix import build_matrix
from rovib.operators.j_operator import JOperator


def _parameters():
    """The inputs for a two mode term with three-dimensional coupling.

    All three axes of zeta are non-zero and a couples the axes, unlike a
    planar molecule such as water. This is what distinguishes the sum over
    :math:`\\beta` of :math:`\\eta_{\\alpha\\beta}\\zeta^\\beta` from, e.g.,
    using :math:`\\zeta^\\alpha`.
    """
    rng = np.random.default_rng(42)
    frequencies = np.array([1.0, 2.0])
    zeta = np.zeros((2, 3, 2))
    zeta[0, :, 1] = [0.3, -0.2, 0.9]
    zeta[1, :, 0] = -zeta[0, :, 1]
    tensor = rng.uniform(-1.0, 1.0, (2, 3, 3))
    a = tensor + tensor.transpose(0, 2, 1)
    gamma = np.array([0.25, 1.0, 0.5])
    return frequencies, zeta, a, gamma


def _angular_momentum(J):
    """The matrices of J_x, J_y, and J_z, ordered by K."""
    K = np.arange(-J, J + 1)
    size = 2 * J + 1
    # <K - 1|J_+|K>, as J_+ lowers K in the molecule-fixed frame.
    plus = np.zeros((size, size))
    for k in range(1, size):
        plus[k - 1, k] = np.sqrt(J * (J + 1) - K[k] * (K[k] - 1))
    x = 0.5 * (plus + plus.T)
    y = (plus - plus.T) / 2j
    return x, y, np.diag(K).astype(complex)


def _brute_force_matrix(frequencies, zeta, a, gamma, vmax, J):
    """The matrix of |Hrovib| from explicit matrices of Q, P, mu, and J.

    The operators are represented in a direct product basis that is large
    enough for the products in |Hrovib|, which have at most 5 operators, to
    be exact within the returned block.
    """
    dim = vmax + 6
    size = dim * dim
    identity = np.eye(dim)
    ladder = np.diag(np.sqrt(np.arange(1, dim)), 1)

    def on_mode(matrix, mode):
        factors = (matrix, identity) if mode == 0 else (identity, matrix)
        return np.kron(*factors)

    q = [
        on_mode(np.sqrt(0.5 / w) * (ladder + ladder.T), n)
        for n, w in enumerate(frequencies)
    ]
    p = [
        on_mode(1j * np.sqrt(0.5 * w) * (ladder.T - ladder), n)
        for n, w in enumerate(frequencies)
    ]
    pi = [
        sum(
            zeta[i, alpha, j] * q[i] @ p[j] for i in range(2) for j in range(2)
        )
        for alpha in range(3)
    ]

    # N.b. the first factor of the Kronecker products is the axis.
    half = np.diag(np.sqrt(2.0 * gamma))
    b = sum(np.kron(half @ a[k] @ half, q[k]) for k in range(2))
    half = np.kron(half, np.eye(size))

    keep = [
        np.ravel_multi_index((i, j), (dim, dim))
        for i in range(vmax + 1)
        for j in range(vmax + 1)
    ]

    def block(matrix):
        return matrix[np.ix_(keep, keep)]

    j_ops = _angular_momentum(J)
    total = 0.0
    for n in range(5):
        c_n = (n + 1) * (-0.5) ** n
        mu = c_n * half @ np.linalg.matrix_power(b, n) @ half
        blocks = mu.reshape(3, size, 3, size)
        for alpha in range(3):
            for beta in range(3):
                mu_ab = blocks[alpha, :, beta, :]
                if n > 0:
                    rotational = j_ops[alpha] @ j_ops[beta]
                    total = total + 0.5 * np.kron(block(mu_ab), rotational)
                if n < 4:
                    coriolis = mu_ab @ pi[beta] + pi[beta] @ mu_ab
                    total = total - 0.5 * np.kron(
                        block(coriolis), j_ops[alpha]
                    )
    return total


class TestRovibrationalHamiltonian(unittest.TestCase):
    def setUp(self):
        self.frequencies, self.zeta, self.a, self.gamma = _parameters()
        self.term = RovibrationalHamiltonian(
            self.frequencies, self.zeta, self.a, self.gamma
        )

    def test_is_hamiltonian_term(self):
        self.assertIsInstance(self.term, HamiltonianTerm)

    def test_state(self):
        np.testing.assert_array_equal(self.term.frequencies, self.frequencies)
        np.testing.assert_array_equal(self.term.coriolis_couplings, self.zeta)
        np.testing.assert_array_equal(self.term.inertia_derivatives, self.a)
        np.testing.assert_array_equal(
            self.term.rotational_constants, self.gamma
        )
        self.assertEqual(self.term.nvib, 2)

    def test_terms(self):
        terms = self.term.terms(RotationalAnsatz(2))
        # 4 orders * 9 pairs of axes, and 4 orders * 3 axes * 2 orderings.
        self.assertEqual(len(terms), 60)
        for _, _, rotational in terms[:36]:
            self.assertEqual(len(rotational), 2)
        for _, _, rotational in terms[36:]:
            self.assertEqual(len(rotational), 1)
            self.assertEqual(rotational[0].J, 2)
        self.assertEqual(terms[1][2], [JOperator(2, 0), JOperator(2, 1)])

    def test_reducers_are_reused(self):
        # vibrational_matrix caches on the reducer.
        once = self.term.terms(RotationalAnsatz(1))
        twice = self.term.terms(RotationalAnsatz(2))
        for first, second in zip(once, twice):
            self.assertIs(first[1], second[1])

    def test_reducers(self):
        terms = self.term.terms(RotationalAnsatz(1))
        eta = np.arange(9.0).reshape(3, 3)
        zeta = np.array([1.0, 10.0, 100.0])
        # The symmetric part of eta[0, 1] = 1 and eta[1, 0] = 3, over 2.
        self.assertAlmostEqual(terms[1][1]([eta]), 1.0)
        # alpha = 1 is the second pair of each order; -(3 + 40 + 500) / 2.
        mu_pi = terms[38][1]
        pi_mu = terms[39][1]
        self.assertAlmostEqual(mu_pi([eta, zeta]), -271.5)
        self.assertAlmostEqual(pi_mu([zeta, eta]), -271.5)

    def test_matches_brute_force(self):
        for vmax, J in ((1, 1), (2, 2)):
            with self.subTest(vmax=vmax, J=J):
                ansatz = RovibrationalAnsatz(2, vmax, J)
                matrix = build_matrix(ansatz, self.term, ansatz)
                np.testing.assert_allclose(
                    matrix,
                    _brute_force_matrix(
                        self.frequencies,
                        self.zeta,
                        self.a,
                        self.gamma,
                        vmax,
                        J,
                    ),
                    atol=1.0e-12,
                )

    def test_zero_for_J0(self):
        ansatz = RovibrationalAnsatz(2, 2, 0)
        matrix = build_matrix(ansatz, self.term, ansatz)
        np.testing.assert_array_equal(matrix, 0.0)

    def test_hermitian(self):
        ansatz = RovibrationalAnsatz(2, 2, 3)
        matrix = build_hamiltonian_matrix(Hamiltonian((self.term,)), ansatz)
        self.assertGreater(np.abs(matrix).max(), 0.0)
        np.testing.assert_allclose(matrix, matrix.conj().T, atol=1.0e-12)

    def test_hermitian_for_asymmetric_inertia_derivatives(self):
        # Normal modes with limited precision give an asymmetric a.
        a = self.a.copy()
        a[0, 0, 1] += 0.01
        term = RovibrationalHamiltonian(
            self.frequencies, self.zeta, a, self.gamma
        )
        ansatz = RovibrationalAnsatz(2, 2, 6)
        matrix = build_matrix(ansatz, term, ansatz)
        np.testing.assert_allclose(matrix, matrix.conj().T, atol=1.0e-12)

    def test_no_coupling(self):
        term = RovibrationalHamiltonian(
            self.frequencies,
            np.zeros((2, 3, 2)),
            np.zeros((2, 3, 3)),
            self.gamma,
        )
        ansatz = RovibrationalAnsatz(2, 1, 2)
        matrix = build_matrix(ansatz, term, ansatz)
        np.testing.assert_array_equal(matrix, 0.0)

    def test_different_J(self):
        bra_ansatz = RovibrationalAnsatz(2, 1, 1)
        ket_ansatz = RovibrationalAnsatz(2, 1, 2)
        block = build_matrix(bra_ansatz, self.term, ket_ansatz)
        self.assertEqual(block.shape, (len(bra_ansatz), len(ket_ansatz)))
        np.testing.assert_array_equal(block, 0.0)

    def test_different_vmax(self):
        bra_ansatz = RovibrationalAnsatz(2, 1, 1)
        ket_ansatz = RovibrationalAnsatz(2, 2, 1)
        block = build_matrix(bra_ansatz, self.term, ket_ansatz)
        full = build_matrix(ket_ansatz, self.term, ket_ansatz)
        rows = [ket_ansatz.index(bra) for bra in bra_ansatz]
        np.testing.assert_allclose(block, full[rows, :], atol=1.0e-12)

    def test_mode_mismatch(self):
        ansatz = RovibrationalAnsatz(2, 1, 1)
        other = RovibrationalAnsatz(3, 1, 1)
        with self.assertRaises(ValueError):
            build_matrix(other, self.term, ansatz)
        with self.assertRaises(ValueError):
            build_matrix(ansatz, self.term, other)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            RovibrationalHamiltonian(
                self.frequencies, np.ones((2, 3, 2)), self.a, self.gamma
            )
        with self.assertRaises(ValueError):
            RovibrationalHamiltonian(
                self.frequencies, self.zeta, np.ones((2, 3)), self.gamma
            )
        with self.assertRaises(ValueError):
            RovibrationalHamiltonian(
                self.frequencies, self.zeta, self.a, [1.0, 0.0, 1.0]
            )
