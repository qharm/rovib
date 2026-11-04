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

"""Defines the RotationalAnsatz class."""

from .rotational_basis_function import RotationalBasisFunction


class RotationalAnsatz:
    """The angular momentum states of a single :math:`J`.

    The space is the states :math:`\\ket{J,K}` with
    :math:`K = -J, \\ldots, J`. Only :math:`J` is stored; basis functions are
    generated on demand.

    Basis functions are ordered by increasing :math:`K`.

    :param J: The total angular momentum quantum number.
    :type J: int
    :raises ValueError: If ``J`` is negative.
    """

    def __init__(self, J):
        if J < 0:
            raise ValueError(f"J must be >= 0, got {J}.")
        self._J = int(J)

    @property
    def J(self):
        """The total angular momentum quantum number.

        :rtype: int
        """
        return self._J

    def __len__(self):
        return 2 * self._J + 1

    def __iter__(self):
        for K in range(-self._J, self._J + 1):
            yield RotationalBasisFunction(self._J, K)

    def __contains__(self, basis_function):
        if not isinstance(basis_function, RotationalBasisFunction):
            return False
        return basis_function.J == self._J

    def __eq__(self, other):
        if not isinstance(other, RotationalAnsatz):
            return NotImplemented
        return self.J == other.J

    def basis_function(self, index):
        """Returns the basis function with the given index.

        :param index: The index of the basis function.
        :type index: int
        :return: The basis function.
        :rtype: RotationalBasisFunction
        :raises IndexError: If ``index`` is not in [0, ``len(self)``).
        """
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} is out of range.")
        return RotationalBasisFunction(self._J, index - self._J)

    def index(self, basis_function):
        """Returns the index of a basis function.

        :param basis_function: The basis function to look up.
        :type basis_function: RotationalBasisFunction
        :return: The index of ``basis_function``.
        :rtype: int
        :raises ValueError: If ``basis_function`` is not in the ansatz.
        """
        if basis_function not in self:
            raise ValueError(f"{basis_function} is not in the ansatz.")
        return basis_function.K + self._J
