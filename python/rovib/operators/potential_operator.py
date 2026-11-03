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

"""Defines the PotentialOperator class."""

import collections
import itertools
import math

import numpy as np

from .normal_mode import NormalMode


class PotentialOperator:
    """One order of the Taylor expansion of the potential.

    For order :math:`m` the operator is

    .. math::

       \\frac{1}{m!}\\sum_{i_1,\\ldots,i_m=1}^{\\Nvib}
          F_{i_1\\ldots i_m}\\qop_{i_1}\\cdots\\qop_{i_m}

    The :math:`\\qop_i` commute, so the sum is restricted to
    :math:`i_1 \\le \\ldots \\le i_m` and the coefficients are scaled to
    account for the terms that were combined.

    Like the other operators made of sums of products of single-mode
    operators (see :ref:`classes`), :meth:`vibrational_operators` returns the
    products and :meth:`coefficient` returns the coefficient of one of them.

    :param force_constants: The force constants :math:`F_{i_1\\ldots i_m}`
        in atomic units. The shape is (``nvib``,) * ``order`` and the tensor
        is assumed to be symmetric.
    :type force_constants: numpy.ndarray
    :param frequencies: The harmonic frequencies |omegavec| in hartree. The
        shape is (``nvib``,).
    :type frequencies: numpy.ndarray
    :raises ValueError: If ``force_constants`` is not of shape
        (``nvib``,) * ``order`` for some ``order`` >= 1.
    """

    def __init__(self, force_constants, frequencies):
        force_constants = np.asarray(force_constants, dtype=float)
        nvib = len(frequencies)
        order = force_constants.ndim
        if order < 1 or force_constants.shape != (nvib,) * order:
            raise ValueError(
                f"Expected force constants of shape (nvib,) * order with "
                f"nvib = {nvib}, got {force_constants.shape}."
            )
        self._order = order
        modes = [
            NormalMode(n, float(omega)) for n, omega in enumerate(frequencies)
        ]
        # N.b. the products and coefficients are computed once, because
        # vibrational_operators is called for every ket.
        self._coefficients = {}
        for idx in itertools.combinations_with_replacement(range(nvib), order):
            powers = collections.Counter(idx).values()
            self._coefficients[tuple(modes[i] for i in idx)] = float(
                force_constants[idx]
            ) / math.prod(math.factorial(p) for p in powers)

    @property
    def order(self):
        """The order of the operator, :math:`m`.

        :rtype: int
        """
        return self._order

    def vibrational_operators(self):
        """The products of normal modes in the operator.

        :return: One tuple of :class:`.NormalMode` objects per
            :math:`i_1 \\le \\ldots \\le i_m`.
        :rtype: collections.abc.Iterable
        """
        return self._coefficients.keys()

    def coefficient(self, operators):
        """The coefficient of a product of normal modes.

        :param operators: A product returned by
            :meth:`vibrational_operators`.
        :type operators: tuple
        :return: The force constant divided by the product of the factorials
            of the powers of each mode.
        :rtype: float
        """
        return self._coefficients[operators]
