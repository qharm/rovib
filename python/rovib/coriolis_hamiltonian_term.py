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

"""Defines the CoriolisHamiltonianTerm class."""

import numpy as np

from .hamiltonian_term import HamiltonianTerm
from .operators.coriolis_operator import CoriolisOperator
from .operators.watson_operator import WatsonOperator

#: The largest order of |muvec| in any term of the Hamiltonian.
_MAX_MU_ORDER = 4


class CoriolisHamiltonianTerm(HamiltonianTerm):
    """Base class for the terms built from |pivec| and |muvec|.

    Holds and validates the inputs of those operators, and builds the
    operators. All inputs are stored as read-only arrays. The harmonic
    frequencies are needed because the matrix elements of :math:`\\qop_i` and
    :math:`\\pop_i` depend on :math:`\\omega_i`.

    :param frequencies: The harmonic frequencies |omegavec| in hartree. The
        shape is (number of vibrational modes,).
    :type frequencies: numpy.ndarray
    :param coriolis_couplings: The Coriolis coupling constants |zetavec|,
        indexed by (:math:`i`, :math:`\\alpha`, :math:`j`). The shape is
        (``nvib``, 3, ``nvib``).
    :type coriolis_couplings: numpy.ndarray
    :param inertia_derivatives: The derivatives of the inertia tensor |avec|
        in atomic units, indexed by (:math:`k`, :math:`\\alpha`,
        :math:`\\beta`). The shape is (``nvib``, 3, 3).
    :type inertia_derivatives: numpy.ndarray
    :param rotational_constants: The rotational constants |gammavec| in
        hartree. These must be in the same axes as ``coriolis_couplings`` and
        ``inertia_derivatives``.
    :type rotational_constants: numpy.ndarray
    :raises ValueError: If ``frequencies`` is not a non-empty vector, if any
        frequency or rotational constant is not positive, if any input has the
        wrong shape, or if ``coriolis_couplings`` is not antisymmetric in
        :math:`i` and :math:`j`.
    """

    def __init__(
        self,
        frequencies,
        coriolis_couplings,
        inertia_derivatives,
        rotational_constants,
    ):
        self._frequencies = np.array(frequencies, dtype=float)
        self._coriolis_couplings = np.array(coriolis_couplings, dtype=float)
        self._inertia_derivatives = np.array(inertia_derivatives, dtype=float)
        self._rotational_constants = np.array(
            rotational_constants, dtype=float
        )

        if self._frequencies.ndim != 1 or self._frequencies.size == 0:
            raise ValueError(
                "Expected a non-empty vector of frequencies, got shape "
                f"{self._frequencies.shape}."
            )
        if np.any(self._frequencies <= 0.0):
            raise ValueError("Harmonic frequencies must be positive.")
        if self._coriolis_couplings.shape != (self.nvib, 3, self.nvib):
            raise ValueError(
                "Expected the Coriolis couplings to have shape "
                f"{(self.nvib, 3, self.nvib)}, got "
                f"{self._coriolis_couplings.shape}."
            )
        if not np.allclose(
            self._coriolis_couplings,
            -self._coriolis_couplings.transpose(2, 1, 0),
        ):
            raise ValueError(
                "The Coriolis couplings must be antisymmetric in the modes."
            )
        if self._inertia_derivatives.shape != (self.nvib, 3, 3):
            raise ValueError(
                "Expected the inertia derivatives to have shape "
                f"{(self.nvib, 3, 3)}, got "
                f"{self._inertia_derivatives.shape}."
            )
        if self._rotational_constants.shape != (3,):
            raise ValueError(
                "Expected three rotational constants, got shape "
                f"{self._rotational_constants.shape}."
            )
        if np.any(self._rotational_constants <= 0.0):
            raise ValueError("Rotational constants must be positive.")

        self._frequencies.setflags(write=False)
        self._coriolis_couplings.setflags(write=False)
        self._inertia_derivatives.setflags(write=False)
        self._rotational_constants.setflags(write=False)

        self._pi = CoriolisOperator(
            self._coriolis_couplings, self._frequencies
        )
        self._mu = tuple(
            WatsonOperator(
                n,
                self._inertia_derivatives,
                self._rotational_constants,
                self._frequencies,
            )
            for n in range(_MAX_MU_ORDER + 1)
        )

    @property
    def frequencies(self):
        """The harmonic frequencies |omegavec| in hartree.

        :rtype: numpy.ndarray
        """
        return self._frequencies

    @property
    def coriolis_couplings(self):
        """The Coriolis coupling constants |zetavec|.

        :rtype: numpy.ndarray
        """
        return self._coriolis_couplings

    @property
    def inertia_derivatives(self):
        """The derivatives of the inertia tensor |avec| in atomic units.

        :rtype: numpy.ndarray
        """
        return self._inertia_derivatives

    @property
    def rotational_constants(self):
        """The rotational constants |gammavec| in hartree.

        :rtype: numpy.ndarray
        """
        return self._rotational_constants

    @property
    def nvib(self):
        """The number of vibrational modes.

        :rtype: int
        """
        return self._frequencies.size
