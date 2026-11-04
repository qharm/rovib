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

"""Defines the HarmonicOperator class."""

import numpy as np

from .number_operator import NumberOperator


class HarmonicOperator:
    """The harmonic oscillator Hamiltonian as a sum of products.

    In atomic units,

    .. math::

       \\HHO = \\sum_{n=1}^{\\Nvib}\\omega_n\\op{n}_n +
          \\frac{1}{2}\\sum_{n=1}^{\\Nvib}\\omega_n

    The zero-point energy is the coefficient of the empty product, i.e., the
    identity.

    Like the other operators made of sums of products of single-mode
    operators (see :ref:`classes`), :meth:`vibrational_operators` returns the
    products and :meth:`coefficient` returns the coefficient of one of them.

    :param frequencies: The harmonic frequencies |omegavec| in hartree. The
        shape is (``nvib``,).
    :type frequencies: numpy.ndarray
    :raises ValueError: If ``frequencies`` is not a vector.
    """

    def __init__(self, frequencies):
        frequencies = np.asarray(frequencies, dtype=float)
        if frequencies.ndim != 1:
            raise ValueError(
                "Expected a vector of frequencies, got shape "
                f"{frequencies.shape}."
            )
        # N.b. the products and coefficients are computed once, because
        # vibrational_operators is called for every ket.
        self._coefficients = {
            (NumberOperator(n),): float(omega)
            for n, omega in enumerate(frequencies)
        }
        self._coefficients[()] = 0.5 * float(frequencies.sum())

    def vibrational_operators(self):
        """The products of number operators in the operator.

        :return: One tuple holding a :class:`.NumberOperator` per mode, and
            the empty tuple, i.e., the identity.
        :rtype: collections.abc.Iterable
        """
        return self._coefficients.keys()

    def coefficient(self, operators):
        """The coefficient of a product of number operators.

        :param operators: A product returned by
            :meth:`vibrational_operators`.
        :type operators: tuple
        :return: The frequency of the mode, or the zero-point energy for the
            empty product, in hartree.
        :rtype: float
        """
        return self._coefficients[operators]
