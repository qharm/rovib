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

"""Defines the ConjugateMomentum class."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ConjugateMomentum:
    """The conjugate momentum :math:`\\pop_n`.

    In atomic units, and in terms of the ladder operators of mode :math:`n`,

    .. math::

       \\pop_n = i\\sqrt{\\frac{\\omega_n}{2}}\\left(a_n^\\dagger - a_n\\right)

    :param mode: The index of the mode, :math:`n`.
    :type mode: int
    :param frequency: The harmonic frequency of the mode in hartree.
    :type frequency: float
    """

    mode: int
    frequency: float

    def act(self, v):
        """Applies the operator to the harmonic oscillator state
        :math:`\\ket{v}`.

        :param v: The vibrational quantum number of the state.
        :type v: int
        :return: The (:math:`v'`, amplitude) pairs of the resulting state in
            atomic units.
        :rtype: list
        """
        scale = math.sqrt(0.5 * self.frequency)
        result = [(v + 1, 1j * scale * math.sqrt(v + 1))]
        if v > 0:
            result.append((v - 1, -1j * scale * math.sqrt(v)))
        return result
