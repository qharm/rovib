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

from rovib import coriolis_couplings, harmonic_analysis, water_molecule


class TestCoriolisCouplings(unittest.TestCase):
    def test_cross_products(self):
        # Two modes on one atom along x and y, plus a mode on a second atom
        # along x.
        L = np.zeros((2, 3, 3))
        L[0, 0, 0] = 1.0
        L[0, 1, 1] = 1.0
        L[1, 0, 2] = 1.0
        zeta = coriolis_couplings(L)
        self.assertEqual(zeta.shape, (3, 3, 3))
        expected = np.zeros((3, 3, 3))
        # x cross y = z
        expected[0, 2, 1] = 1.0
        expected[1, 2, 0] = -1.0
        np.testing.assert_allclose(zeta, expected)

    def test_sums_over_atoms(self):
        L = np.zeros((2, 3, 2))
        L[:, 0, 0] = [1.0, 2.0]
        L[:, 1, 1] = [3.0, 4.0]
        zeta = coriolis_couplings(L)
        self.assertAlmostEqual(zeta[0, 2, 1], 1.0 * 3.0 + 2.0 * 4.0)

    def test_antisymmetric(self):
        rng = np.random.default_rng(42)
        zeta = coriolis_couplings(rng.uniform(-1.0, 1.0, (4, 3, 5)))
        self.assertEqual(zeta.shape, (5, 3, 5))
        np.testing.assert_allclose(zeta, -zeta.transpose(2, 1, 0))

    def test_water(self):
        # Water is planar, with x perpendicular to the plane, so only the x
        # component is non-zero.
        L, _ = harmonic_analysis(water_molecule())
        zeta = coriolis_couplings(L)
        np.testing.assert_array_equal(zeta[:, 1:, :], 0.0)
        # The bend and antisymmetric stretch are the strongly coupled pair.
        self.assertAlmostEqual(abs(zeta[1, 0, 2]), 1.0, places=3)
        self.assertLess(abs(zeta[0, 0, 2]), 0.01)
        self.assertEqual(zeta[0, 0, 1], 0.0)
