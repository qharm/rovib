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

"""Defines the RovibrationalBasisFunction class."""

from dataclasses import dataclass

from .rotational_basis_function import RotationalBasisFunction
from .vibrational_basis_function import VibrationalBasisFunction


@dataclass(frozen=True, init=False)
class RovibrationalBasisFunction:
    """A rovibrational basis function :math:`\\ket{\\vvec,J,K}`.

    The basis function is the product of a :class:`.VibrationalBasisFunction`
    and a :class:`.RotationalBasisFunction`. Instances are immutable and
    hashable.

    :param v: The vibrational quantum number of each mode.
    :type v: tuple of int
    :param J: The total angular momentum quantum number.
    :type J: int
    :param K: The projection of the total angular momentum on to the
        molecule-fixed z-axis.
    :type K: int
    :raises ValueError: If any vibrational quantum number is negative, if
        ``J`` is negative, or if :math:`|K| > J`.
    """

    vibrational: VibrationalBasisFunction
    rotational: RotationalBasisFunction

    def __init__(self, v, J, K):
        vibrational = VibrationalBasisFunction(v)
        rotational = RotationalBasisFunction(J, K)
        object.__setattr__(self, "vibrational", vibrational)
        object.__setattr__(self, "rotational", rotational)

    @property
    def v(self):
        """The vibrational quantum number of each mode.

        :rtype: tuple of int
        """
        return self.vibrational.v

    @property
    def J(self):
        """The total angular momentum quantum number.

        :rtype: int
        """
        return self.rotational.J

    @property
    def K(self):
        """The projection of |jvec| on to the molecule-fixed z-axis.

        :rtype: int
        """
        return self.rotational.K

    @property
    def nvib(self):
        """The number of vibrational modes.

        :rtype: int
        """
        return self.vibrational.nvib
