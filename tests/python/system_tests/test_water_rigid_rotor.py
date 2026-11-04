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
"""Reproduces the "RR + HO" data of Figs. 2(b) and 2(c) of
doi:10.1063/5.0333947.

The reference data are from the paper's supporting information
(``RovibData_v2/Fig2/Fig2b.txt`` and ``Fig2c.txt``). They are the levels of
the (100) and (001) vibrational states, relative to the J = 0 ground state,
and the sign of the wavefunction under exchange of hydrogens,
Pex = (-1)^(v3 + K) of each level. Pex=-1 means the wavefunction is
antisymmetric under exchange of hydrogens and thus the spin state must be a
singlet. Pex=1 is the opposite (spin must be a triplet).
"""

import unittest

import numpy as np

from rovib import (
    HarmonicHamiltonian,
    RigidRotorHamiltonian,
    RovibrationalAnsatz,
    build_hamiltonian_matrix,
    dominant_basis_function,
    harmonic_analysis,
    hartree_to_wavenumber,
    rotational_constants,
    water_molecule,
)

#: The vibrational states in Figs. 2(b) and 2(c). Came from guess and checking
STATES = ((1, 0, 0), (0, 0, 1))

#: The paper's (DeltaE in cm^-1, Pex) pairs, keyed by J.
LEVELS = {
    0: (
        (3830.8700, 1),
        (3940.4600, -1),
    ),
    3: (
        (3969.4548, -1),
        (3974.5288, 1),
        (4004.8018, -1),
        (4036.0868, 1),
        (4041.8930, -1),
        (4079.0448, 1),
        (4084.1188, -1),
        (4112.8222, 1),
        (4113.0205, -1),
        (4114.3918, 1),
        (4145.6768, -1),
        (4151.4830, 1),
        (4222.4122, -1),
        (4222.6105, 1),
    ),
}

#: The rotational constants the paper used are about 1.5e-4 (relative)
#: smaller than those from the geometry in its Appendix B, which shifts the
#: J = 3 levels by up to 0.043 cm^-1. With the paper's rotational constants
#: the levels agree to 1e-4 cm^-1, i.e., to the printed precision.
TOL = 0.05


def _eigensystem(water, J):
    """Returns the energies in cm^-1, eigenvectors, and ansatz of HRR + HHO."""
    _, frequencies = harmonic_analysis(water)
    hamiltonian = HarmonicHamiltonian(frequencies) + RigidRotorHamiltonian(
        rotational_constants(water)
    )
    ansatz = RovibrationalAnsatz(water.nvib, vmax=1, J=J)
    matrix = build_hamiltonian_matrix(hamiltonian, ansatz)
    energies, eigenvectors = np.linalg.eigh(matrix)
    return hartree_to_wavenumber(energies), eigenvectors, ansatz


def _levels(J):
    """Returns the (DeltaE in cm^-1, Pex) pairs of the levels in STATES."""
    water = water_molecule()
    ground_state = _eigensystem(water, 0)[0][0]
    energies, eigenvectors, ansatz = _eigensystem(water, J)
    levels = []
    for i, energy in enumerate(energies):
        bf = dominant_basis_function(eigenvectors[:, i], ansatz)
        if bf.v in STATES:
            levels.append((energy - ground_state, (-1) ** (bf.v[2] + bf.K)))
    return levels


class TestWaterRigidRotor(unittest.TestCase):
    def test_fig2_rr_ho(self):
        for J, expected in LEVELS.items():
            levels = _levels(J)
            self.assertEqual(len(levels), len(expected))
            for (energy, pex), (ref_energy, ref_pex) in zip(levels, expected):
                with self.subTest(J=J, level=ref_energy):
                    self.assertAlmostEqual(energy, ref_energy, delta=TOL)
                    self.assertEqual(pex, ref_pex)
