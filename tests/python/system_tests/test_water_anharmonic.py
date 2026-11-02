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
"""Reproduces the "RR + anHO" data of Figs. 2(b) and 2(c) of
doi:10.1063/5.0333947.

The reference data are from the paper's supporting information
(``RovibData_v2/Fig2/Fig2b.txt`` and ``Fig2c.txt``). They are the levels of
HRR + HHO + Vanharm with vmax = 3, relative to the J = 0 ground state, in
the windows 3700 cm^-1 <= DeltaE <= 4000 cm^-1 for J = 0
and 3800 cm^-1 <= DeltaE <= 4300 cm^-1 for J = 3, and the sign of the
wavefunction under exchange of hydrogens, Pex = (-1)^(v3 + K), of each
level.
"""

import unittest

import numpy as np

from rovib import (
    AnharmonicPotential,
    HarmonicHamiltonian,
    RigidRotorHamiltonian,
    RovibrationalAnsatz,
    anharmonic_force_constants,
    build_hamiltonian_matrix,
    dominant_basis_function,
    harmonic_analysis,
    hartree_to_wavenumber,
    rotational_constants,
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
        (3716.3448, 1),
        (3782.6483, -1),
    ),
    3: (
        (3854.9295, -1),
        (3860.0036, 1),
        (3890.2766, -1),
        (3921.2331, 1),
        (3921.5616, 1),
        (3926.3071, -1),
        (3927.3678, -1),
        (3956.5801, 1),
        (3987.8652, -1),
        (3993.6713, 1),
        (3998.2970, 1),
        (3998.4953, -1),
        (4064.6006, -1),
        (4064.7988, 1),
    ),
}

#: As in the "RR + HO" test, the paper's rotational constants are slightly
#: smaller than those from the geometry in its Appendix B, which shifts the
#: J = 3 levels by up to 0.042 cm^-1. The J = 0 levels agree to 0.002 cm^-1.
TOL = 0.05


def _hamiltonian(water):
    """Returns HRR + HHO + Vanharm for water."""
    _, frequencies = harmonic_analysis(water)
    cubic, quartic = anharmonic_force_constants(water)
    return (
        HarmonicHamiltonian(frequencies)
        + RigidRotorHamiltonian(rotational_constants(water))
        + AnharmonicPotential(frequencies, cubic, quartic)
    )


def _eigensystem(hamiltonian, J):
    """Returns the energies in cm^-1, eigenvectors, and ansatz for J."""
    ansatz = RovibrationalAnsatz(3, VMAX, J)
    matrix = build_hamiltonian_matrix(hamiltonian, ansatz)
    energies, eigenvectors = np.linalg.eigh(matrix)
    return hartree_to_wavenumber(energies), eigenvectors, ansatz


def _levels(J):
    """Returns the (DeltaE in cm^-1, Pex) pairs of the levels in WINDOWS."""
    hamiltonian = _hamiltonian(water_molecule())
    ground_state = _eigensystem(hamiltonian, 0)[0][0]
    energies, eigenvectors, ansatz = _eigensystem(hamiltonian, J)
    lower, upper = WINDOWS[J]
    levels = []
    for i, energy in enumerate(energies - ground_state):
        if lower <= energy <= upper:
            bf = dominant_basis_function(eigenvectors[:, i], ansatz)
            levels.append((energy, (-1) ** (bf.v[2] + bf.K)))
    return levels


class TestWaterAnharmonic(unittest.TestCase):
    def test_fig2_rr_anho(self):
        for J, expected in LEVELS.items():
            levels = _levels(J)
            self.assertEqual(len(levels), len(expected))
            for (energy, pex), (ref_energy, ref_pex) in zip(levels, expected):
                with self.subTest(J=J, level=ref_energy):
                    self.assertAlmostEqual(energy, ref_energy, delta=TOL)
                    self.assertEqual(pex, ref_pex)
