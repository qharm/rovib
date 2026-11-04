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

"""Defines the NumberOperator class."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NumberOperator:
    """The number operator :math:`\\op{n}_n = a_n^\\dagger a_n`.

    :param mode: The index of the mode, :math:`n`.
    :type mode: int
    """

    mode: int

    def act(self, v):
        """Applies the operator to the harmonic oscillator state
        :math:`\\ket{v}`.

        :param v: The vibrational quantum number of the state.
        :type v: int
        :return: The (:math:`v'`, amplitude) pairs of the resulting state.
        :rtype: list
        """
        return [(v, float(v))]
