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
import unittest

import numpy as np

from rovib import (
    AnharmonicPotential,
    RovibrationalAnsatz,
    RovibrationalBasisFunction,
)
from rovib.anharmonic_potential_elements import (
    anharmonic_matrix_element,
    anharmonic_non_zero_indices,
)


def _symmetrize(tensor):
    """Averages ``tensor`` over all permutations of its indices."""
    permutations = list(itertools.permutations(range(tensor.ndim)))
    return sum(tensor.transpose(p) for p in permutations) / len(permutations)


def _random_potential(frequencies, seed=42):
    """An AnharmonicPotential with random, symmetric force constants."""
    rng = np.random.default_rng(seed)
    nvib = len(frequencies)
    cubic = _symmetrize(rng.uniform(-1.0, 1.0, (nvib,) * 3))
    quartic = _symmetrize(rng.uniform(-1.0, 1.0, (nvib,) * 4))
    return AnharmonicPotential(frequencies, cubic, quartic)


def _brute_force_matrix(term, vmax):
    """The matrix of ``term`` in a direct product basis of HO states.

    The basis has :math:`0 \\le v_n \\le v_{\\max}`, ordered as in
    :class:`.RovibrationalAnsatz` with :math:`J = 0`. The operator is built
    from the unrestricted sums over the force constants, with each
    :math:`\\qop_n` represented in a larger basis so that products of up to
    four of them are exact within the returned block.
    """
    dim = vmax + 3
    ladder = np.diag(np.sqrt(np.arange(1, dim)), 1)
    identity = np.eye(dim)
    q = []
    for n, omega in enumerate(term.frequencies):
        factors = [identity] * term.nvib
        factors[n] = np.sqrt(0.5 / omega) * (ladder + ladder.T)
        q_n = factors[0]
        for factor in factors[1:]:
            q_n = np.kron(q_n, factor)
        q.append(q_n)

    modes = range(term.nvib)
    v = np.zeros_like(q[0])
    for i, j, k in itertools.product(modes, repeat=3):
        v += term.cubic[i, j, k] * q[i] @ q[j] @ q[k] / 6.0
    for i, j, k, m in itertools.product(modes, repeat=4):
        v += term.quartic[i, j, k, m] * q[i] @ q[j] @ q[k] @ q[m] / 24.0

    keep = [
        np.ravel_multi_index(idx, (dim,) * term.nvib)
        for idx in itertools.product(range(vmax + 1), repeat=term.nvib)
    ]
    return v[np.ix_(keep, keep)]


def _element_matrix(term, ansatz):
    """The matrix of ``term`` built element by element."""
    return np.array(
        [
            [anharmonic_matrix_element(bra, term, ket) for ket in ansatz]
            for bra in ansatz
        ]
    )


class TestAnharmonicMatrixElement(unittest.TestCase):
    def setUp(self):
        self.term = _random_potential([1.0, 2.0, 3.0])

    def test_single_mode_cubic(self):
        # V = Q^3, so <v+1|V|v> = (1 / (2 omega))^(3/2) * 3 (v+1)^(3/2).
        term = AnharmonicPotential([2.0], [[[6.0]]], np.zeros((1, 1, 1, 1)))
        bra = RovibrationalBasisFunction((3,), 0, 0)
        ket = RovibrationalBasisFunction((2,), 0, 0)
        value = anharmonic_matrix_element(bra, term, ket)
        self.assertIsInstance(value, complex)
        self.assertAlmostEqual(value, 0.25**1.5 * 3.0 * 3.0**1.5)
        self.assertAlmostEqual(
            anharmonic_matrix_element(ket, term, bra), value
        )

    def test_single_mode_quartic(self):
        # V = Q^4, so <v|V|v> = (1 / (2 omega))^2 * (6v^2 + 6v + 3).
        term = AnharmonicPotential([2.0], np.zeros((1, 1, 1)), [[[[24.0]]]])
        bf = RovibrationalBasisFunction((2,), 0, 0)
        value = anharmonic_matrix_element(bf, term, bf)
        self.assertAlmostEqual(value, 0.25**2 * 39.0)

    def test_matches_brute_force(self):
        vmax = 2
        ansatz = RovibrationalAnsatz(self.term.nvib, vmax, 0)
        np.testing.assert_allclose(
            _element_matrix(self.term, ansatz),
            _brute_force_matrix(self.term, vmax),
            atol=1.0e-12,
        )

    def test_matches_brute_force_one_mode(self):
        term = _random_potential([1.5])
        vmax = 6
        ansatz = RovibrationalAnsatz(1, vmax, 0)
        np.testing.assert_allclose(
            _element_matrix(term, ansatz),
            _brute_force_matrix(term, vmax),
            atol=1.0e-12,
        )

    def test_diagonal_in_J_and_K(self):
        ket = RovibrationalBasisFunction((1, 0, 2), 2, 0)
        # Without the J, K check these would couple through the quartic
        # terms.
        for J, K in ((2, 1), (2, 2), (3, 0)):
            with self.subTest(J=J, K=K):
                bra = RovibrationalBasisFunction((1, 0, 2), J, K)
                self.assertEqual(
                    anharmonic_matrix_element(bra, self.term, ket), 0.0
                )

    def test_independent_of_J_and_K(self):
        bra = RovibrationalBasisFunction((1, 1, 1), 0, 0)
        ket = RovibrationalBasisFunction((0, 1, 2), 0, 0)
        expected = anharmonic_matrix_element(bra, self.term, ket)
        self.assertNotEqual(expected, 0.0)
        bra = RovibrationalBasisFunction((1, 1, 1), 2, -1)
        ket = RovibrationalBasisFunction((0, 1, 2), 2, -1)
        self.assertEqual(
            anharmonic_matrix_element(bra, self.term, ket), expected
        )

    def test_too_many_quanta(self):
        bra = RovibrationalBasisFunction((3, 2, 0), 0, 0)
        ket = RovibrationalBasisFunction((0, 0, 0), 0, 0)
        self.assertEqual(anharmonic_matrix_element(bra, self.term, ket), 0.0)

    def test_zero_force_constants(self):
        term = AnharmonicPotential(
            [1.0, 2.0], np.zeros((2,) * 3), np.zeros((2,) * 4)
        )
        bf = RovibrationalBasisFunction((1, 1), 0, 0)
        self.assertEqual(anharmonic_matrix_element(bf, term, bf), 0.0)


class TestAnharmonicNonZeroIndices(unittest.TestCase):
    def setUp(self):
        self.term = _random_potential([1.0, 2.0, 3.0])
        self.ansatz = RovibrationalAnsatz(3, 2, 1)

    def test_all_non_zero_elements_generated(self):
        pairs = list(
            anharmonic_non_zero_indices(self.ansatz, self.term, self.ansatz)
        )
        self.assertEqual(len(set(pairs)), len(pairs))
        pairs = set(pairs)
        for bra, ket in itertools.product(self.ansatz, repeat=2):
            if (bra, ket) not in pairs:
                self.assertEqual(
                    anharmonic_matrix_element(bra, self.term, ket), 0.0
                )

    def test_selection_rules(self):
        pairs = anharmonic_non_zero_indices(
            self.ansatz, self.term, self.ansatz
        )
        for bra, ket in pairs:
            self.assertEqual((bra.J, bra.K), (ket.J, ket.K))
            delta_v = np.subtract(bra.v, ket.v)
            self.assertLessEqual(np.abs(delta_v).sum(), 4)

    def test_only_cubic(self):
        # Only g_000 is non-zero, so mode 0 changes by 1 or 3 and the other
        # modes do not change.
        cubic = np.zeros((3, 3, 3))
        cubic[0, 0, 0] = 1.0
        term = AnharmonicPotential([1.0, 2.0, 3.0], cubic, np.zeros((3,) * 4))
        ansatz = RovibrationalAnsatz(3, 3, 0)
        ket = RovibrationalBasisFunction((1, 2, 0), 0, 0)
        pairs = anharmonic_non_zero_indices(ansatz, term, ansatz)
        bras = [bra.v for bra, k in pairs if k == ket]
        self.assertEqual(sorted(bras), [(0, 2, 0), (2, 2, 0)])

    def test_zero_force_constants(self):
        term = AnharmonicPotential(
            [1.0, 2.0, 3.0], np.zeros((3,) * 3), np.zeros((3,) * 4)
        )
        pairs = anharmonic_non_zero_indices(self.ansatz, term, self.ansatz)
        self.assertEqual(list(pairs), [])

    def test_different_spaces(self):
        bra_space = RovibrationalAnsatz(3, 1, 1)
        pairs = list(
            anharmonic_non_zero_indices(bra_space, self.term, self.ansatz)
        )
        self.assertGreater(len(pairs), 0)
        for bra, ket in pairs:
            self.assertIn(bra, bra_space)
            self.assertIn(ket, self.ansatz)

    def test_different_J(self):
        bra_space = RovibrationalAnsatz(3, 2, 0)
        pairs = anharmonic_non_zero_indices(bra_space, self.term, self.ansatz)
        self.assertEqual(list(pairs), [])

    def test_mode_mismatch(self):
        other = RovibrationalAnsatz(2, 2, 1)
        with self.assertRaises(ValueError):
            list(anharmonic_non_zero_indices(other, self.term, self.ansatz))
        with self.assertRaises(ValueError):
            list(anharmonic_non_zero_indices(self.ansatz, self.term, other))
