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

"""Defines the HarmonicHamiltonian class."""

import numpy as np

from .hamiltonian_term import HamiltonianTerm
from .operators.harmonic_operator import HarmonicOperator


def _harmonic_coefficient(coefficients):
    """Reducer for |HHO|, which has a single factor."""
    return coefficients[0]


class HarmonicHamiltonian(HamiltonianTerm):
    """The harmonic oscillator Hamiltonian |HHO|.

    The frequencies are stored as a read-only array.

    :param frequencies: The harmonic frequencies |omegavec| in hartree. The
        shape is (number of vibrational modes,).
    :type frequencies: numpy.ndarray
    :raises ValueError: If ``frequencies`` is not a non-empty vector, or if
        any frequency is not positive.
    """

    def __init__(self, frequencies):
        self._frequencies = np.array(frequencies, dtype=float)

        if self._frequencies.ndim != 1 or self._frequencies.size == 0:
            raise ValueError(
                "Expected a non-empty vector of frequencies, got shape "
                f"{self._frequencies.shape}."
            )
        if np.any(self._frequencies <= 0.0):
            raise ValueError("Harmonic frequencies must be positive.")

        self._frequencies.setflags(write=False)

        # N.b. the inputs are read-only, so the terms only need to be built
        # once.
        self._terms = (
            ([HarmonicOperator(self._frequencies)], _harmonic_coefficient),
        )

    @property
    def frequencies(self):
        """The harmonic frequencies |omegavec| in hartree.

        :rtype: numpy.ndarray
        """
        return self._frequencies

    @property
    def nvib(self):
        """The number of vibrational modes.

        :rtype: int
        """
        return self._frequencies.size

    def terms(self):
        """The number operators and zero-point energy of |HHO|.

        :return: The (operators, reducer) pair of |HHO|, in the form expected
            by :func:`.vibrational_matrix`.
        :rtype: tuple
        """
        return self._terms
