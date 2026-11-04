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
    HarmonicHamiltonian,
    Molecule,
    RigidRotorHamiltonian,
    RovibrationalAnsatz,
    RovibrationalBasisFunction,
    VibrationalCoriolisHamiltonian,
    anharmonic_force_constants,
    build_hamiltonian_matrix,
    coriolis_couplings,
    dominant_basis_function,
    harmonic_analysis,
    hartree_to_wavenumber,
    inertia_derivatives,
    rotational_constants,
    rovibrational_energies,
    water_molecule,
)
from rovib.constants import HARTREE_TO_CM


class TestRovibrationalEnergies(unittest.TestCase):
    def test_returns(self):
        energies, eigenvectors = rovibrational_energies(
            water_molecule(), vmax=1, J=1
        )
        ansatz = RovibrationalAnsatz(3, 1, 1)
        self.assertEqual(energies.shape, (len(ansatz),))
        self.assertEqual(eigenvectors.shape, (len(ansatz), len(ansatz)))
        self.assertTrue(np.all(np.diff(energies) >= 0.0))

    def test_hamiltonian_terms(self):
        water = water_molecule()
        normal_modes, frequencies = harmonic_analysis(water)
        cubic, quartic = anharmonic_force_constants(water)
        gamma = rotational_constants(water)
        hamiltonian = (
            HarmonicHamiltonian(frequencies)
            + RigidRotorHamiltonian(gamma)
            + AnharmonicPotential(frequencies, cubic, quartic)
            + VibrationalCoriolisHamiltonian(
                frequencies,
                coriolis_couplings(normal_modes),
                inertia_derivatives(water, normal_modes),
                gamma,
            )
        )
        ansatz = RovibrationalAnsatz(water.nvib, 1, 1)
        matrix = build_hamiltonian_matrix(hamiltonian, ansatz)
        energies, _ = rovibrational_energies(water, vmax=1, J=1)
        np.testing.assert_allclose(energies, np.linalg.eigvalsh(matrix))

    def test_rigid_rotor(self):
        # With vmax = 0 the vibrational terms are the same constant for all
        # J, so relative to J = 0 the J = 1 levels are the J = 1 rigid rotor
        # levels, which are sums of two rotational constants.
        water = water_molecule()
        energies, _ = rovibrational_energies(water, vmax=0, J=1)
        zero_point = rovibrational_energies(water, vmax=0, J=0)[0][0]
        x, y, z = rotational_constants(water)
        np.testing.assert_allclose(
            energies - zero_point, sorted([x + y, x + z, y + z])
        )

    def test_not_water(self):
        water = water_molecule()
        ammonia_like = Molecule(("N", "H", "H"), water.coords, water.masses)
        with self.assertRaises(ValueError):
            rovibrational_energies(ammonia_like, vmax=1)


class TestDominantBasisFunction(unittest.TestCase):
    def test_dominant(self):
        ansatz = RovibrationalAnsatz(2, 1, 0)
        vector = np.array([0.1, -0.9, 0.3, 0.2])
        self.assertEqual(
            dominant_basis_function(vector, ansatz),
            RovibrationalBasisFunction((0, 1), 0, 0),
        )


class TestHartreeToWavenumber(unittest.TestCase):
    def test_conversion(self):
        self.assertEqual(hartree_to_wavenumber(1.0), HARTREE_TO_CM)
        np.testing.assert_allclose(
            hartree_to_wavenumber([0.5, 2.0]),
            [0.5 * HARTREE_TO_CM, 2.0 * HARTREE_TO_CM],
        )
