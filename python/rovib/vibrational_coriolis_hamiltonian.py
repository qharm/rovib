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

"""Defines the VibrationalCoriolisHamiltonian class."""

import numpy as np

from .coriolis_hamiltonian_term import CoriolisHamiltonianTerm

#: The largest order of |muvec| in the Coriolis part of |HvibCor|.
_MAX_CORIOLIS_ORDER = 2

#: The largest order of |muvec| in the Watson part of |HvibCor|.
_MAX_WATSON_ORDER = 4


def _coriolis_coefficient(coefficients):
    """Reducer for :math:`\\frac{1}{2}\\pivec^T\\muvec^{(n)}\\pivec`."""
    zeta_left, eta, zeta_right = coefficients
    return 0.5 * float(zeta_left @ eta @ zeta_right)


def _watson_coefficient(coefficients):
    """Reducer for :math:`-\\frac{1}{8}\\tr(\\muvec^{(n)})`."""
    return -0.125 * float(np.trace(coefficients[0]))


class VibrationalCoriolisHamiltonian(CoriolisHamiltonianTerm):
    """The vibrational Coriolis Hamiltonian |HvibCor|.

    In atomic units,

    .. math::

       \\HvibCor = \\frac{1}{2}\\sum_{n=0}^{2}\\pivec^T\\muvec^{(n)}\\pivec -
                   \\frac{1}{8}\\sum_{n=0}^{4}\\tr(\\muvec^{(n)})

    The harmonic frequencies are needed because the matrix elements of
    :math:`\\qop_i` and :math:`\\pop_i` depend on :math:`\\omega_i`. All
    inputs are stored as read-only arrays. See :ref:`vibration_coriolis` for
    the matrix elements.

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
        super().__init__(
            frequencies,
            coriolis_couplings,
            inertia_derivatives,
            rotational_constants,
        )

        # N.b. the inputs are read-only, so the terms only need to be built
        # once.
        pi, mu = self._pi, self._mu
        self._terms = tuple(
            ([pi, mu[n], pi], _coriolis_coefficient, [])
            for n in range(_MAX_CORIOLIS_ORDER + 1)
        ) + tuple(
            ([mu[n]], _watson_coefficient, [])
            for n in range(_MAX_WATSON_ORDER + 1)
        )

    def terms(self, kets):
        """The Coriolis and Watson parts of |HvibCor|.

        :param kets: The rotational part of the space the term acts on. The
            term is purely vibrational, so this is not used.
        :type kets: RotationalAnsatz
        :return: The products of
            :math:`\\frac{1}{2}\\pivec^T\\muvec^{(n)}\\pivec` for
            :math:`n = 0, 1, 2`, followed by those of
            :math:`-\\frac{1}{8}\\tr(\\muvec^{(n)})` for
            :math:`n = 0, \\ldots, 4`, in the form described by
            :meth:`.HamiltonianTerm.terms`.
        :rtype: tuple
        """
        return self._terms
