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
import io
import unittest
from unittest import mock

import numpy as np

from rovib import (
    Molecule,
    RovibrationalAnsatz,
    RovibrationalBasisFunction,
    dominant_basis_function,
    encoded_harmonic_hamiltonian,
    hartree_to_wavenumber,
    print_encoded_harmonic_hamiltonian,
    rovibrational_energies,
    water_molecule,
)
from rovib.constants import HARTREE_TO_CM
from rovib.water_params import water_harmonic_frequencies


class TestRovibrationalEnergies(unittest.TestCase):
    def test_returns(self):
        energies, eigenvectors = rovibrational_energies(
            water_molecule(), vmax=1, J=1
        )
        ansatz = RovibrationalAnsatz(3, 1, 1)
        self.assertEqual(energies.shape, (len(ansatz),))
        self.assertEqual(eigenvectors.shape, (len(ansatz), len(ansatz)))
        self.assertTrue(np.all(np.diff(energies) >= 0.0))

    def test_not_water(self):
        water = water_molecule()
        ammonia_like = Molecule(("N", "H", "H"), water.coords, water.masses)
        with self.assertRaises(ValueError):
            rovibrational_energies(ammonia_like, vmax=1)


class TestEncodedHarmonicHamiltonian(unittest.TestCase):
    def test_water(self):
        encoded = encoded_harmonic_hamiltonian(water_molecule(), vmax=3)
        self.assertEqual(len(encoded), 7)
        self.assertAlmostEqual(
            encoded["IIIIII"],
            2 * sum(water_harmonic_frequencies()),
        )

    def test_water_pauli_sum_reproduces_all_harmonic_energies(self):
        frequencies = water_harmonic_frequencies()
        encoded = encoded_harmonic_hamiltonian(water_molecule(), vmax=3)

        for occupations in itertools.product(range(4), repeat=3):
            # Each mode uses two qubits.  Bit 0 is the least-significant
            # qubit within a mode, while Pauli strings are printed high to
            # low, matching the convention used by the encoder.
            state = sum(occupation << (2 * mode)
                        for mode, occupation in enumerate(occupations))
            encoded_energy = 0.0
            for pauli, coefficient in encoded.items():
                eigenvalue = np.prod(
                    [
                        1.0
                        if operator == "I"
                        else (-1.0 if (state >> qubit) & 1 else 1.0)
                        for qubit, operator in enumerate(reversed(pauli))
                    ]
                )
                encoded_energy += coefficient * eigenvalue

            expected_energy = np.dot(frequencies, np.add(occupations, 0.5))
            self.assertAlmostEqual(encoded_energy, expected_energy)

    def test_print_report(self):
        output = io.StringIO()
        with mock.patch("sys.stdout", output):
            print_encoded_harmonic_hamiltonian(
                water_molecule(), vmax=3, nstates=5
            )
        report = output.getvalue()
        self.assertIn("Pauli coefficients (hartree):", report)
        self.assertIn("IIIIII", report)
        self.assertIn("Lowest harmonic energies:", report)
        self.assertIn("v=(0, 0, 0)", report)


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
