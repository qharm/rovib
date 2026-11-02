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
"""The top-level driver: from a molecule to rovibrational energies."""

import itertools

import numpy as np

from .constants import HARTREE_TO_CM
from .hamiltonian import Hamiltonian
from .hamiltonian_matrix import build_hamiltonian_matrix
from .harmonic_hamiltonian import HarmonicHamiltonian
from .rovibrational_ansatz import RovibrationalAnsatz
from .water_params import harmonic_analysis


def rovibrational_energies(molecule, vmax, J=0):
    """Computes the rovibrational energies of a molecule.

    The Hamiltonian is currently only the harmonic oscillator Hamiltonian
    |HHO|, and the parameters are hard-coded for water. The eigenvectors are
    expressed in ``RovibrationalAnsatz(molecule.nvib, vmax, J)``.

    :param molecule: The molecule, at its minimum energy geometry.
    :type molecule: Molecule
    :param vmax: The maximum vibrational quantum number of each mode.
    :type vmax: int
    :param J: The total angular momentum quantum number.
    :type J: int
    :return: The energies in hartree (in ascending order) and the
        corresponding eigenvectors (as columns).
    :rtype: tuple of (numpy.ndarray, numpy.ndarray)
    :raises ValueError: If ``molecule`` is not water.
    """
    # The normal modes are not needed until the Coriolis terms are added.
    _, frequencies = harmonic_analysis(molecule)

    H_harm = HarmonicHamiltonian(frequencies)

    hamiltonian = Hamiltonian((H_harm, ))
    ansatz = RovibrationalAnsatz(molecule.nvib, vmax, J)
    matrix = build_hamiltonian_matrix(hamiltonian, ansatz)
    energies, eigenvectors = np.linalg.eigh(matrix)
    return energies, eigenvectors


def encode_harmonic_hamiltonian(frequencies, vmax):
    r"""Encode the harmonic Hamiltonian as a weighted Pauli sum.

    A binary encoding is used for each vibrational occupation number, and the
    full qubit-space matrix is expanded in the Pauli basis.  The matrix agrees
    with the harmonic Hamiltonian on the truncated basis states and is zero on
    unused binary states, as in Eq. (22) of the supplied paper.

    The returned dictionary maps Pauli strings to their real coefficients in
    hartree.  Characters are ordered from the highest-numbered qubit to qubit
    zero, matching the paper's convention
    ``P = sigma_(Nq-1) \u2297 ... \u2297 sigma_0``.  Qubits are grouped by
    vibrational mode, with bit 0 as the least-significant qubit of each mode.
    The identity string is included explicitly when its coefficient is
    non-zero.

    :param frequencies: The harmonic frequencies in hartree.
    :type frequencies: array-like
    :param vmax: The largest occupation retained for every mode.
    :type vmax: int
    :return: A mapping from Pauli strings to coefficients in hartree.
    :rtype: dict[str, float]
    :raises ValueError: If ``frequencies`` or ``vmax`` is invalid.
    """
    term = HarmonicHamiltonian(frequencies)
    if not isinstance(vmax, (int, np.integer)) or isinstance(vmax, bool):
        raise ValueError(f"vmax must be a non-negative integer, got {vmax!r}.")
    if vmax < 0:
        raise ValueError(f"vmax must be a non-negative integer, got {vmax}.")
    vmax = int(vmax)

    # ceil(log2(vmax + 1)), computed exactly for integer input.
    qubits_per_mode = (vmax).bit_length()
    nqubits = term.nvib * qubits_per_mode
    dimension = 2**nqubits
    diagonal = np.zeros(dimension)
    occupation_mask = (1 << qubits_per_mode) - 1
    for state in range(dimension):
        occupations = tuple(
            (state >> (mode * qubits_per_mode)) & occupation_mask
            for mode in range(term.nvib)
        )
        if all(occupation <= vmax for occupation in occupations):
            diagonal[state] = np.dot(term.frequencies, np.add(occupations, 0.5))

    pauli_sum = {}
    for mask in range(dimension):
        parity = np.array(
            [
                -1.0 if (mask & state).bit_count() % 2 else 1.0
                for state in range(dimension)
            ]
        )
        coefficient = float(np.dot(diagonal, parity) / dimension)
        if not np.isclose(coefficient, 0.0):
            pauli = "".join(
                "Z" if mask & (1 << qubit) else "I"
                for qubit in reversed(range(nqubits))
            )
            pauli_sum[pauli] = coefficient

    return pauli_sum


def encoded_harmonic_hamiltonian(molecule, vmax):
    """Returns the encoded harmonic Hamiltonian for a molecule.

    Harmonic analysis supplies the molecule's frequencies, which are then
    encoded using :func:`encode_harmonic_hamiltonian`.  Currently
    :func:`~rovib.water_params.harmonic_analysis` provides hard-coded
    parameters for water and rejects other molecules.

    :param molecule: The molecule at its minimum-energy geometry.
    :type molecule: Molecule
    :param vmax: The largest occupation retained for every vibrational mode.
    :type vmax: int
    :return: A mapping from Pauli strings to coefficients in hartree.
    :rtype: dict[str, float]
    :raises ValueError: If the molecule is not supported or ``vmax`` is
        invalid.
    """
    _, frequencies = harmonic_analysis(molecule)
    return encode_harmonic_hamiltonian(frequencies, vmax)


def print_encoded_harmonic_hamiltonian(molecule, vmax, nstates=5):
    """Print Pauli coefficients and the lowest encoded harmonic energies.

    The Pauli coefficients are printed in hartree.  The energies are printed
    in cm^-1, together with their excitation energies relative to the ground
    state.  ``nstates`` includes the ground state.

    :param molecule: The molecule at its minimum-energy geometry.
    :type molecule: Molecule
    :param vmax: The largest occupation retained for every vibrational mode.
    :type vmax: int
    :param nstates: Number of lowest states to print, including the ground
        state.
    :type nstates: int
    :raises ValueError: If the molecule, ``vmax``, or ``nstates`` is invalid.
    """
    if not isinstance(nstates, (int, np.integer)) or isinstance(nstates, bool):
        raise ValueError(
            f"nstates must be a positive integer, got {nstates!r}."
        )
    if nstates < 1:
        raise ValueError(f"nstates must be a positive integer, got {nstates}.")

    _, frequencies = harmonic_analysis(molecule)
    pauli_sum = encode_harmonic_hamiltonian(frequencies, vmax)
    vmax = int(vmax)
    print("Pauli coefficients (hartree):")
    for pauli, coefficient in pauli_sum.items():
        print(f"{coefficient:+.12e} * {pauli}")

    qubits_per_mode = vmax.bit_length()
    states = []
    for occupations in itertools.product(
        range(vmax + 1), repeat=len(frequencies)
    ):
        state = sum(
            occupation << (qubits_per_mode * mode)
            for mode, occupation in enumerate(occupations)
        )
        energy = 0.0
        for pauli, coefficient in pauli_sum.items():
            eigenvalue = np.prod(
                [
                    1.0
                    if operator == "I"
                    else (-1.0 if (state >> qubit) & 1 else 1.0)
                    for qubit, operator in enumerate(reversed(pauli))
                ]
            )
            energy += coefficient * eigenvalue
        states.append((float(energy), occupations))
    states.sort(key=lambda state: state[0])
    ground_energy = states[0][0]
    print("Lowest harmonic energies:")
    for energy, occupations in states[:nstates]:
        energy_cm = float(hartree_to_wavenumber(energy))
        excitation_cm = float(hartree_to_wavenumber(energy - ground_energy))
        print(
            f"v={occupations}: E={energy_cm:.4f} cm^-1, "
            f"excitation={excitation_cm:.4f} cm^-1"
        )


def dominant_basis_function(eigenvector, ansatz):
    """Returns the basis function that contributes most to an eigenvector.

    :param eigenvector: The eigenvector's coefficients in ``ansatz``.
    :type eigenvector: numpy.ndarray
    :param ansatz: The ansatz the eigenvector is expressed in.
    :type ansatz: RovibrationalAnsatz
    :return: The basis function with the largest coefficient magnitude.
    :rtype: RovibrationalBasisFunction
    """

    # N.b. np.argmax(np.abs(...)) is the index of the largest absolute value.
    return ansatz.basis_function(int(np.argmax(np.abs(eigenvector))))


def hartree_to_wavenumber(energy):
    """Converts energies from hartree to wavenumbers.

    :param energy: The energies in hartree.
    :type energy: numpy.ndarray
    :return: The energies in cm^-1.
    :rtype: numpy.ndarray
    """
    return np.asarray(energy) * HARTREE_TO_CM
