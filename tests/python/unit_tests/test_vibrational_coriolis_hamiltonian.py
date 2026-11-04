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
    VibrationalCoriolisHamiltonian,
    build_hamiltonian_matrix,
)
from rovib.operators.coriolis_operator import CoriolisOperator
from rovib.operators.watson_operator import WatsonOperator


def _parameters():
    """The inputs for a two mode term with three-dimensional coupling."""
    rng = np.random.default_rng(42)
    frequencies = np.array([1.0, 2.0])
    zeta = np.zeros((2, 3, 2))
    zeta[0, :, 1] = [0.3, -0.2, 0.9]
    zeta[1, :, 0] = -zeta[0, :, 1]
    tensor = rng.uniform(-1.0, 1.0, (2, 3, 3))
    a = tensor + tensor.transpose(0, 2, 1)
    gamma = np.array([0.25, 1.0, 0.5])
    return frequencies, zeta, a, gamma


def _brute_force_matrix(frequencies, zeta, a, gamma, vmax):
    """The matrix of |HvibCor| from explicit matrices of Q, P, and mu.

    The operators are represented in a direct product basis that is large
    enough for the products in |HvibCor|, which have at most 6 operators, to
    be exact within the returned block.
    """
    dim = vmax + 7
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

    total = np.zeros((size, size), dtype=complex)
    for n in range(5):
        c_n = (n + 1) * (-0.5) ** n
        mu = c_n * half @ np.linalg.matrix_power(b, n) @ half
        blocks = mu.reshape(3, size, 3, size)
        for alpha in range(3):
            total -= 0.125 * blocks[alpha, :, alpha, :]
            if n > 2:
                continue
            for beta in range(3):
                total += 0.5 * pi[alpha] @ blocks[alpha, :, beta, :] @ pi[beta]

    keep = [
        np.ravel_multi_index((i, j), (dim, dim))
        for i in range(vmax + 1)
        for j in range(vmax + 1)
    ]
    return total[np.ix_(keep, keep)]


class TestVibrationalCoriolisHamiltonian(unittest.TestCase):
    def setUp(self):
        self.frequencies, self.zeta, self.a, self.gamma = _parameters()
        self.term = VibrationalCoriolisHamiltonian(
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
        terms = self.term.terms(RotationalAnsatz(1))
        self.assertEqual(len(terms), 8)
        for _, _, rotational in terms:
            self.assertEqual(rotational, [])
        for n, (operators, _, _) in enumerate(terms[:3]):
            with self.subTest(coriolis=n):
                self.assertEqual(len(operators), 3)
                self.assertIsInstance(operators[0], CoriolisOperator)
                self.assertIsInstance(operators[1], WatsonOperator)
                self.assertIsInstance(operators[2], CoriolisOperator)
                self.assertEqual(operators[1].order, n)
        for n, (operators, _, _) in enumerate(terms[3:]):
            with self.subTest(watson=n):
                self.assertEqual(len(operators), 1)
                self.assertIsInstance(operators[0], WatsonOperator)
                self.assertEqual(operators[0].order, n)

    def test_reducers(self):
        terms = self.term.terms(RotationalAnsatz(1))
        coriolis = terms[0][1]
        watson = terms[3][1]
        left = np.array([1.0, 2.0, 3.0])
        eta = np.diag([1.0, 10.0, 100.0])
        right = np.array([3.0, 2.0, 1.0])
        # (3 + 40 + 300) / 2
        self.assertAlmostEqual(coriolis([left, eta, right]), 171.5)
        self.assertAlmostEqual(watson([eta]), -111.0 / 8.0)

    def test_matches_brute_force(self):
        ansatz = RovibrationalAnsatz(2, 2, 0)
        matrix = build_hamiltonian_matrix(Hamiltonian((self.term,)), ansatz)
        np.testing.assert_allclose(
            matrix,
            _brute_force_matrix(
                self.frequencies, self.zeta, self.a, self.gamma, 2
            ),
            atol=1.0e-12,
        )

    def test_hermitian(self):
        ansatz = RovibrationalAnsatz(2, 3, 1)
        matrix = build_hamiltonian_matrix(Hamiltonian((self.term,)), ansatz)
        self.assertGreater(np.abs(matrix).max(), 0.0)
        np.testing.assert_allclose(matrix, matrix.conj().T, atol=1.0e-12)

    def test_no_coupling(self):
        # With zeta = 0 and a = 0, only the constant -tr(mu^(0)) / 8 is left.
        term = VibrationalCoriolisHamiltonian(
            self.frequencies,
            np.zeros((2, 3, 2)),
            np.zeros((2, 3, 3)),
            self.gamma,
        )
        ansatz = RovibrationalAnsatz(2, 1, 0)
        matrix = build_hamiltonian_matrix(Hamiltonian((term,)), ansatz)
        expected = -0.125 * 2.0 * self.gamma.sum()
        np.testing.assert_allclose(matrix, expected * np.eye(len(ansatz)))

    def test_state_is_read_only(self):
        with self.assertRaises(ValueError):
            self.term.frequencies[0] = 1.0
        with self.assertRaises(ValueError):
            self.term.coriolis_couplings[0, 0, 0] = 1.0
        with self.assertRaises(ValueError):
            self.term.inertia_derivatives[0, 0, 0] = 1.0
        with self.assertRaises(ValueError):
            self.term.rotational_constants[0] = 1.0

    def test_input_is_copied(self):
        expected = self.a[0, 0, 0]
        self.frequencies[0] = 100.0
        self.zeta[0, 0, 1] = 100.0
        self.a[0, 0, 0] = 100.0
        self.gamma[0] = 100.0
        self.assertEqual(self.term.frequencies[0], 1.0)
        self.assertEqual(self.term.coriolis_couplings[0, 0, 1], 0.3)
        self.assertEqual(self.term.inertia_derivatives[0, 0, 0], expected)
        self.assertEqual(self.term.rotational_constants[0], 0.25)

    def test_bad_frequencies(self):
        for frequencies in (np.ones((2, 1)), [], [1.0, 0.0], [1.0, -2.0]):
            with self.subTest(frequencies=frequencies):
                with self.assertRaises(ValueError):
                    VibrationalCoriolisHamiltonian(
                        frequencies, self.zeta, self.a, self.gamma
                    )

    def test_bad_coriolis_couplings_shape(self):
        for shape in ((2, 2), (2, 2, 2), (3, 3, 3)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    VibrationalCoriolisHamiltonian(
                        self.frequencies,
                        np.zeros(shape),
                        self.a,
                        self.gamma,
                    )

    def test_symmetric_coriolis_couplings(self):
        zeta = np.zeros((2, 3, 2))
        zeta[0, 0, 1] = 1.0
        with self.assertRaises(ValueError):
            VibrationalCoriolisHamiltonian(
                self.frequencies, zeta, self.a, self.gamma
            )

    def test_bad_inertia_derivatives_shape(self):
        for shape in ((3, 3), (2, 3, 2), (3, 3, 3)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    VibrationalCoriolisHamiltonian(
                        self.frequencies,
                        self.zeta,
                        np.zeros(shape),
                        self.gamma,
                    )

    def test_bad_rotational_constants(self):
        for gamma in ([1.0, 2.0], [1.0, 0.0, 2.0], [1.0, -1.0, 2.0]):
            with self.subTest(gamma=gamma):
                with self.assertRaises(ValueError):
                    VibrationalCoriolisHamiltonian(
                        self.frequencies, self.zeta, self.a, gamma
                    )
