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

"""Defines the JOperator class."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class JOperator:
    """A molecule-fixed component of the angular momentum |jvec|.

    In atomic units, and in terms of the ladder operators,

    .. math::

       \\jop_x = \\frac{1}{2}\\left(\\jop_+ + \\jop_-\\right)\\qquad
       \\jop_y = \\frac{1}{2i}\\left(\\jop_+ - \\jop_-\\right)

    where :math:`\\jop_+` lowers :math:`K` and :math:`\\jop_-` raises it. See
    :ref:`angular_wavefunction`.

    :param J: The total angular momentum quantum number.
    :type J: int
    :param projection: The component of |jvec|: 0, 1, or 2 for x, y, or z.
        These are the indices of the rotational constants |gammavec|.
    :type projection: int
    :raises ValueError: If ``J`` is negative, or if ``projection`` is not 0,
        1, or 2.
    """

    J: int
    projection: int

    def __post_init__(self):
        if self.J < 0:
            raise ValueError(f"J must be >= 0, got {self.J}.")
        if self.projection not in (0, 1, 2):
            raise ValueError(
                f"projection must be 0, 1, or 2, got {self.projection}."
            )

    def act(self, K):
        """Applies the operator to the angular momentum state
        :math:`\\ket{J,K}`.

        :param K: The projection of |jvec| on to the molecule-fixed z-axis.
        :type K: int
        :return: The (:math:`K'`, amplitude) pairs of the resulting state in
            atomic units.
        :rtype: list
        """
        if self.projection == 2:
            return [(K, float(K))]
        # The coefficients of J_+ and J_- in J_x or J_y.
        plus, minus = (0.5, 0.5) if self.projection == 0 else (-0.5j, 0.5j)
        J_squared = self.J * (self.J + 1)
        result = []
        if K > -self.J:
            result.append((K - 1, plus * math.sqrt(J_squared - K * (K - 1))))
        if K < self.J:
            result.append((K + 1, minus * math.sqrt(J_squared - K * (K + 1))))
        return result
