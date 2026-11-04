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

"""Defines the VibrationalAnsatz class."""

import itertools

from .vibrational_basis_function import VibrationalBasisFunction


class VibrationalAnsatz:
    """The direct product of harmonic oscillator states.

    The space has :math:`0 \\le v_k \\le v_{\\max}` for every mode. Only the
    truncation parameters are stored; basis functions are generated on demand.

    Basis functions are ordered with the first mode's quantum number varying
    slowest.

    :param nvib: The number of vibrational modes.
    :type nvib: int
    :param vmax: The maximum vibrational quantum number of each mode.
    :type vmax: int
    :raises ValueError: If ``nvib`` is less than 1 or if ``vmax`` is negative.
    """

    def __init__(self, nvib, vmax):
        if nvib < 1:
            raise ValueError(f"nvib must be >= 1, got {nvib}.")
        if vmax < 0:
            raise ValueError(f"vmax must be >= 0, got {vmax}.")
        self._nvib = int(nvib)
        self._vmax = int(vmax)

    @property
    def nvib(self):
        """The number of vibrational modes.

        :rtype: int
        """
        return self._nvib

    @property
    def vmax(self):
        """The maximum vibrational quantum number of each mode.

        :rtype: int
        """
        return self._vmax

    def __len__(self):
        return (self._vmax + 1) ** self._nvib

    def __iter__(self):
        quanta = range(self._vmax + 1)
        for v in itertools.product(quanta, repeat=self._nvib):
            yield VibrationalBasisFunction(v)

    def __contains__(self, basis_function):
        if not isinstance(basis_function, VibrationalBasisFunction):
            return False
        return basis_function.nvib == self._nvib and all(
            vk <= self._vmax for vk in basis_function.v
        )

    def __eq__(self, other):
        if not isinstance(other, VibrationalAnsatz):
            return NotImplemented
        return (self.nvib, self.vmax) == (other.nvib, other.vmax)

    def basis_function(self, index):
        """Returns the basis function with the given index.

        :param index: The index of the basis function.
        :type index: int
        :return: The basis function.
        :rtype: VibrationalBasisFunction
        :raises IndexError: If ``index`` is not in [0, ``len(self)``).
        """
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} is out of range.")
        v = []
        for _ in range(self._nvib):
            index, vk = divmod(index, self._vmax + 1)
            v.append(vk)
        return VibrationalBasisFunction(tuple(reversed(v)))

    def index(self, basis_function):
        """Returns the index of a basis function.

        :param basis_function: The basis function to look up.
        :type basis_function: VibrationalBasisFunction
        :return: The index of ``basis_function``.
        :rtype: int
        :raises ValueError: If ``basis_function`` is not in the ansatz.
        """
        if basis_function not in self:
            raise ValueError(f"{basis_function} is not in the ansatz.")
        index = 0
        for vk in basis_function.v:
            index = index * (self._vmax + 1) + vk
        return index
