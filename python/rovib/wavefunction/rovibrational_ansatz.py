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

"""Defines the RovibrationalAnsatz class."""

from .rotational_ansatz import RotationalAnsatz
from .rovibrational_basis_function import RovibrationalBasisFunction
from .vibrational_ansatz import VibrationalAnsatz


class RovibrationalAnsatz:
    """The space in which the Hamiltonian is diagonalized.

    The space is the direct product of a :class:`.VibrationalAnsatz` and a
    :class:`.RotationalAnsatz`, i.e., harmonic oscillator states with
    :math:`0 \\le v_k \\le v_{\\max}` for every mode, and the angular
    momentum states :math:`\\ket{J,K}` with :math:`K = -J, \\ldots, J` for a
    single :math:`J`. Only the truncation parameters are stored; basis
    functions are generated on demand.

    Basis functions are ordered with the first mode's quantum number varying
    slowest and :math:`K` varying fastest.

    :param nvib: The number of vibrational modes.
    :type nvib: int
    :param vmax: The maximum vibrational quantum number of each mode.
    :type vmax: int
    :param J: The total angular momentum quantum number.
    :type J: int
    :raises ValueError: If ``nvib`` is less than 1 or if ``vmax`` or ``J`` is
        negative.
    """

    def __init__(self, nvib, vmax, J):
        self._vibrational = VibrationalAnsatz(nvib, vmax)
        self._rotational = RotationalAnsatz(J)

    @property
    def vibrational(self):
        """The vibrational part of the space.

        :rtype: VibrationalAnsatz
        """
        return self._vibrational

    @property
    def rotational(self):
        """The rotational part of the space.

        :rtype: RotationalAnsatz
        """
        return self._rotational

    @property
    def nvib(self):
        """The number of vibrational modes.

        :rtype: int
        """
        return self._vibrational.nvib

    @property
    def vmax(self):
        """The maximum vibrational quantum number of each mode.

        :rtype: int
        """
        return self._vibrational.vmax

    @property
    def J(self):
        """The total angular momentum quantum number.

        :rtype: int
        """
        return self._rotational.J

    def __len__(self):
        return len(self._vibrational) * len(self._rotational)

    def __iter__(self):
        for vibrational in self._vibrational:
            for rotational in self._rotational:
                yield RovibrationalBasisFunction(
                    vibrational.v, rotational.J, rotational.K
                )

    def __contains__(self, basis_function):
        if not isinstance(basis_function, RovibrationalBasisFunction):
            return False
        return (
            basis_function.vibrational in self._vibrational
            and basis_function.rotational in self._rotational
        )

    def __eq__(self, other):
        if not isinstance(other, RovibrationalAnsatz):
            return NotImplemented
        return (self.vibrational, self.rotational) == (
            other.vibrational,
            other.rotational,
        )

    def basis_function(self, index):
        """Returns the basis function with the given index.

        :param index: The index of the basis function.
        :type index: int
        :return: The basis function.
        :rtype: RovibrationalBasisFunction
        :raises IndexError: If ``index`` is not in [0, ``len(self)``).
        """
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} is out of range.")
        v_index, K_index = divmod(index, len(self._rotational))
        vibrational = self._vibrational.basis_function(v_index)
        rotational = self._rotational.basis_function(K_index)
        return RovibrationalBasisFunction(
            vibrational.v, rotational.J, rotational.K
        )

    def index(self, basis_function):
        """Returns the index of a basis function.

        :param basis_function: The basis function to look up.
        :type basis_function: RovibrationalBasisFunction
        :return: The index of ``basis_function``.
        :rtype: int
        :raises ValueError: If ``basis_function`` is not in the ansatz.
        """
        if basis_function not in self:
            raise ValueError(f"{basis_function} is not in the ansatz.")
        v_index = self._vibrational.index(basis_function.vibrational)
        K_index = self._rotational.index(basis_function.rotational)
        return v_index * len(self._rotational) + K_index
