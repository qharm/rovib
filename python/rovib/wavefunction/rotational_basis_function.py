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

"""Defines the RotationalBasisFunction class."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RotationalBasisFunction:
    """A rotational basis function :math:`\\ket{J,K}`.

    The basis function is an angular momentum eigenstate. Instances are
    immutable and hashable.

    :param J: The total angular momentum quantum number.
    :type J: int
    :param K: The projection of the total angular momentum on to the
        molecule-fixed z-axis.
    :type K: int
    :raises ValueError: If ``J`` is negative, or if :math:`|K| > J`.
    """

    J: int
    K: int

    def __post_init__(self):
        if self.J < 0:
            raise ValueError(f"J must be >= 0, got {self.J}.")
        if abs(self.K) > self.J:
            raise ValueError(f"|K| must be <= J, got J={self.J}, K={self.K}.")
