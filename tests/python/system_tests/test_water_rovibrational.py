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

"""Tests the full Hamiltonian for water against doi:10.1063/5.0333947.

The reference data are from the paper's supporting information
(``RovibData_v2/Fig2/Fig2b.txt`` and ``Fig2c.txt``). They are the levels of
HRR + HHO + Vanharm + HvibCor + Hrovib with vmax = 3, relative to the J = 0
ground state, in the windows 3700 cm^-1 <= DeltaE <= 4000 cm^-1 for J = 0
and 3800 cm^-1 <= DeltaE <= 4300 cm^-1 for J = 3, and the sign of the
wavefunction under exchange of hydrogens, Pex = (-1)^(v3 + K), of each
level.
"""

import unittest

import numpy as np

from rovib import (
    RovibrationalAnsatz,
    dominant_basis_function,
    hartree_to_wavenumber,
    rovibrational_energies,
    water_molecule,
)

#: The maximum vibrational quantum number used in Fig. 2.
VMAX = 3

#: The DeltaE window, in cm^-1, of each panel, keyed by J.
WINDOWS = {
    0: (3700.0, 4000.0),
    3: (3800.0, 4300.0),
}

#: The paper's (DeltaE in cm^-1, Pex) pairs, keyed by J.
LEVELS = {
    0: (
        (3715.7808, 1),
        (3797.0109, -1),
    ),
    3: (
        (3867.5179, -1),
        (3873.1381, 1),
        (3908.5778, -1),
        (3943.3898, 1),
        (3947.3002, -1),
        (3949.9196, 1),
        (3958.5367, -1),
        (3991.7317, 1),
        (4019.7851, -1),
        (4024.2761, 1),
        (4035.0248, -1),
        (4037.7776, 1),
        (4107.4475, -1),
        (4107.7370, 1),
    ),
}

#: As in the other system tests, the paper's rotational constants are slightly
#: smaller than those from the geometry in its Appendix B, which shifts the
#: J = 3 levels by up to 0.054 cm^-1 here. The J = 0 levels agree to
#: 0.002 cm^-1.
TOL = 0.06


def _levels(J):
    """Returns the (DeltaE in cm^-1, Pex) pairs of the levels in WINDOWS."""
    water = water_molecule()
    ground_state = rovibrational_energies(water, VMAX, 0)[0][0]
    energies, eigenvectors = rovibrational_energies(water, VMAX, J)
    ansatz = RovibrationalAnsatz(water.nvib, VMAX, J)
    lower, upper = WINDOWS[J]
    levels = []
    for i, energy in enumerate(hartree_to_wavenumber(energies - ground_state)):
        if lower <= energy <= upper:
            bf = dominant_basis_function(eigenvectors[:, i], ansatz)
            levels.append((energy, (-1) ** (bf.v[2] + bf.K)))
    return levels


class TestWaterRovibrational(unittest.TestCase):
    def test_fig2_full_hamiltonian(self):
        for J, expected in LEVELS.items():
            levels = _levels(J)
            self.assertEqual(len(levels), len(expected))
            for (energy, pex), (ref_energy, ref_pex) in zip(levels, expected):
                with self.subTest(J=J, level=ref_energy):
                    self.assertAlmostEqual(energy, ref_energy, delta=TOL)
                    self.assertEqual(pex, ref_pex)

    def test_hermitian_at_high_J(self):
        # The limited precision of the normal modes makes the inertia
        # derivatives slightly asymmetric. Unless that is accounted for, the
        # driver raises at the largest J in the paper, J = 6.
        energies, _ = rovibrational_energies(water_molecule(), VMAX, 6)
        self.assertTrue(np.all(np.isfinite(energies)))
