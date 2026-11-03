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

"""Defines the AnharmonicPotential class."""

import itertools

import numpy as np

from .hamiltonian_term import HamiltonianTerm
from .operators.potential_operator import PotentialOperator


def _only_coefficient(coefficients):
    """Reducer for an operator with a single factor."""
    return coefficients[0]


def _check_force_constants(force_constants, nvib, order, name):
    """Checks that ``force_constants`` is a symmetric tensor over the modes.

    :raises ValueError: If ``force_constants`` does not have shape
        (``nvib``,) * ``order``, or if it is not symmetric under permutation
        of its indices.
    """
    if force_constants.shape != (nvib,) * order:
        raise ValueError(
            f"Expected the {name} force constants to have shape "
            f"{(nvib,) * order}, got {force_constants.shape}."
        )
    for permutation in itertools.permutations(range(order)):
        if not np.allclose(
            force_constants, force_constants.transpose(permutation)
        ):
            raise ValueError(
                f"The {name} force constants must be symmetric under "
                "permutation of their indices."
            )


class AnharmonicPotential(HamiltonianTerm):
    """The anharmonic potential |Vanharm|.

    The force constants are the derivatives of the potential with respect to
    the mass-weighted normal coordinates, i.e.,

    .. math::

       \\Vanharm = \\frac{1}{3!}\\sum_{ijk=1}^{\\Nvib}
                   g_{ijk}\\qop_i\\qop_j\\qop_k +
                   \\frac{1}{4!}\\sum_{ijkl=1}^{\\Nvib}
                   h_{ijkl}\\qop_i\\qop_j\\qop_k\\qop_l

    The harmonic frequencies are needed because the matrix elements of
    :math:`\\qop_i` depend on :math:`\\omega_i`. All inputs are stored as
    read-only arrays. See :ref:`anharmonic_potential` for the matrix
    elements.

    :param frequencies: The harmonic frequencies |omegavec| in hartree. The
        shape is (number of vibrational modes,).
    :type frequencies: numpy.ndarray
    :param cubic: The cubic force constants :math:`g_{ijk}` in atomic units.
        The shape is (``nvib``, ``nvib``, ``nvib``).
    :type cubic: numpy.ndarray
    :param quartic: The quartic force constants :math:`h_{ijkl}` in atomic
        units. The shape is (``nvib``, ``nvib``, ``nvib``, ``nvib``).
    :type quartic: numpy.ndarray
    :raises ValueError: If ``frequencies`` is not a non-empty vector, if any
        frequency is not positive, or if ``cubic`` or ``quartic`` has the
        wrong shape or is not symmetric under permutation of its indices.
    """

    def __init__(self, frequencies, cubic, quartic):
        self._frequencies = np.array(frequencies, dtype=float)
        self._cubic = np.array(cubic, dtype=float)
        self._quartic = np.array(quartic, dtype=float)

        if self._frequencies.ndim != 1 or self._frequencies.size == 0:
            raise ValueError(
                "Expected a non-empty vector of frequencies, got shape "
                f"{self._frequencies.shape}."
            )
        if np.any(self._frequencies <= 0.0):
            raise ValueError("Harmonic frequencies must be positive.")
        _check_force_constants(self._cubic, self.nvib, 3, "cubic")
        _check_force_constants(self._quartic, self.nvib, 4, "quartic")

        self._frequencies.setflags(write=False)
        self._cubic.setflags(write=False)
        self._quartic.setflags(write=False)

        # N.b. the inputs are read-only, so the terms only need to be built
        # once.
        self._terms = tuple(
            (
                [PotentialOperator(force_constants, self._frequencies)],
                _only_coefficient,
            )
            for force_constants in (self._cubic, self._quartic)
        )

    @property
    def frequencies(self):
        """The harmonic frequencies |omegavec| in hartree.

        :rtype: numpy.ndarray
        """
        return self._frequencies

    @property
    def cubic(self):
        """The cubic force constants :math:`g_{ijk}` in atomic units.

        :rtype: numpy.ndarray
        """
        return self._cubic

    @property
    def quartic(self):
        """The quartic force constants :math:`h_{ijkl}` in atomic units.

        :rtype: numpy.ndarray
        """
        return self._quartic

    @property
    def nvib(self):
        """The number of vibrational modes.

        :rtype: int
        """
        return self._frequencies.size

    def terms(self):
        """The cubic and quartic parts of |Vanharm|.

        Each part is a single :class:`.PotentialOperator`, so each product
        has one factor and the reducer returns that factor's coefficient.

        :return: The (operators, reducer) pairs of the cubic and quartic
            parts, in that order, in the form expected by
            :func:`.vibrational_matrix`.
        :rtype: tuple
        """
        return self._terms
