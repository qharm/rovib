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

"""Defines the WatsonOperator class."""

import itertools

import numpy as np

from .normal_mode import NormalMode


class WatsonOperator:
    """One order of the Taylor expansion of the Watson mu tensor |muvec|.

    For order :math:`n` the operator is

    .. math::

       \\muop^{(n)}_{\\alpha\\beta} = \\sum_{\\midx_n}
          \\eta^{\\midx_n}_{\\alpha\\beta}\\qop_{\\midx_n}

    so :meth:`coefficient` returns a matrix, with one element per component.
    The matrices are combined with the coefficients of the other factors of
    the operator by the reducer (see :func:`.vibrational_matrix`). Unlike
    :class:`.PotentialOperator`, the sum runs over all multi-indices because
    :math:`\\eta^{\\midx_n}_{\\alpha\\beta}` is not symmetric under
    permutation of the multi-index. See :ref:`watson_mu_tensor` for its
    definition.

    Like the other operators made of sums of products of single-mode
    operators (see :ref:`classes`), :meth:`vibrational_operators` returns the
    products and :meth:`coefficient` returns the coefficient of one of them.

    :param order: The order of the operator, :math:`n`.
    :type order: int
    :param inertia_derivatives: The derivatives of the inertia tensor |avec|,
        indexed by (:math:`k`, :math:`\\alpha`, :math:`\\beta`). The shape
        is (``nvib``, 3, 3).
    :type inertia_derivatives: numpy.ndarray
    :param rotational_constants: The rotational constants |gammavec| in
        hartree. These must be in the same axes as ``inertia_derivatives``.
    :type rotational_constants: numpy.ndarray
    :param frequencies: The harmonic frequencies |omegavec| in hartree. The
        shape is (``nvib``,).
    :type frequencies: numpy.ndarray
    :raises ValueError: If ``order`` is negative, if
        ``inertia_derivatives`` is not of shape (``nvib``, 3, 3), or if
        ``rotational_constants`` is not of shape (3,).
    """

    def __init__(
        self, order, inertia_derivatives, rotational_constants, frequencies
    ):
        inertia_derivatives = np.asarray(inertia_derivatives, dtype=float)
        rotational_constants = np.asarray(rotational_constants, dtype=float)
        nvib = len(frequencies)
        if order < 0:
            raise ValueError(f"The order must not be negative, got {order}.")
        if inertia_derivatives.shape != (nvib, 3, 3):
            raise ValueError(
                f"Expected inertia derivatives of shape {(nvib, 3, 3)}, got "
                f"{inertia_derivatives.shape}."
            )
        if rotational_constants.shape != (3,):
            raise ValueError(
                "Expected three rotational constants, got shape "
                f"{rotational_constants.shape}."
            )
        self._order = order
        # N.b. gamma_alpha = 1 / (2 I^0_alphaalpha) in atomic units.
        inverse_moments = 2.0 * rotational_constants
        xi = inverse_moments[np.newaxis, :, np.newaxis] * inertia_derivatives
        c_n = (order + 1) * (-0.5) ** order
        modes = [
            NormalMode(n, float(omega)) for n, omega in enumerate(frequencies)
        ]
        # N.b. the products and coefficients are computed once, because
        # vibrational_operators is called for every ket.
        self._coefficients = {}
        for idx in itertools.product(range(nvib), repeat=order):
            eta = c_n * np.eye(3)
            for k in idx:
                eta = eta @ xi[k]
            eta = eta * inverse_moments
            eta.setflags(write=False)
            self._coefficients[tuple(modes[k] for k in idx)] = eta

    @property
    def order(self):
        """The order of the operator, :math:`n`.

        :rtype: int
        """
        return self._order

    def vibrational_operators(self):
        """The products of normal modes in the operator.

        :return: One tuple of :class:`.NormalMode` objects per multi-index.
            For order 0 the only product is the empty tuple, i.e., the
            identity.
        :rtype: collections.abc.Iterable
        """
        return self._coefficients.keys()

    def coefficient(self, operators):
        """The coefficient of a product of normal modes.

        :param operators: A product returned by
            :meth:`vibrational_operators`.
        :type operators: tuple
        :return: The matrix :math:`\\eta^{\\midx_n}_{\\alpha\\beta}` in atomic
            units. The shape is (3, 3).
        :rtype: numpy.ndarray
        """
        return self._coefficients[operators]
