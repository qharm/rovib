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

"""Tests HRR + HHO + Vanharm + HvibCor for water against doi:10.1063/5.0333947.

The reference data are from the paper's supporting information
(``RovibData_v2/Fig2/Fig2b.txt`` and ``Fig2c.txt``). They are the levels of
HRR + HHO + Vanharm + HvibCor with vmax = 3, relative to the J = 0 ground
state, in the windows 3700 cm^-1 <= DeltaE <= 4000 cm^-1 for J = 0
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
    VibrationalCoriolisHamiltonian,
    anharmonic_force_constants,
    build_hamiltonian_matrix,
    coriolis_couplings,
    dominant_basis_function,
    harmonic_analysis,
    hartree_to_wavenumber,
    inertia_derivatives,
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
        (3715.7808, 1),
        (3797.0109, -1),
    ),
    3: (
        (3854.3656, -1),
        (3859.4396, 1),
        (3889.7126, -1),
        (3920.9977, 1),
        (3926.8038, -1),
        (3935.5957, 1),
        (3940.6697, -1),
        (3970.9427, 1),
        (3997.7331, 1),
        (3997.9314, -1),
        (4002.2277, -1),
        (4008.0339, 1),
        (4078.9631, -1),
        (4079.1614, 1),
    ),
}

#: As in the "RR + anHO" test, the paper's rotational constants are slightly
#: smaller than those from the geometry in its Appendix B, which shifts the
#: J = 3 levels by up to 0.045 cm^-1. The J = 0 levels agree to 0.002 cm^-1.
TOL = 0.05


def _hamiltonian(water):
    """Returns HRR + HHO + Vanharm + HvibCor for water."""
    normal_modes, frequencies = harmonic_analysis(water)
    cubic, quartic = anharmonic_force_constants(water)
    gamma = rotational_constants(water)
    return (
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


class TestWaterVibrationalCoriolis(unittest.TestCase):
    def test_fig2_rr_anho_vibcor(self):
        for J, expected in LEVELS.items():
            levels = _levels(J)
            self.assertEqual(len(levels), len(expected))
            for (energy, pex), (ref_energy, ref_pex) in zip(levels, expected):
                with self.subTest(J=J, level=ref_energy):
                    self.assertAlmostEqual(energy, ref_energy, delta=TOL)
                    self.assertEqual(pex, ref_pex)
