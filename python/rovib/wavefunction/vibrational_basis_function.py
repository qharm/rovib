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

"""Defines the VibrationalBasisFunction class."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VibrationalBasisFunction:
    """A vibrational basis function :math:`\\ket{\\vvec}`.

    The basis function is the product of one harmonic oscillator
    eigenfunction per vibrational mode. Instances are immutable and hashable.

    :param v: The vibrational quantum number of each mode.
    :type v: tuple of int
    :raises ValueError: If any vibrational quantum number is negative.
    """

    v: tuple

    def __post_init__(self):
        object.__setattr__(self, "v", tuple(int(vk) for vk in self.v))
        if any(vk < 0 for vk in self.v):
            raise ValueError(f"Quantum numbers must be >= 0, got {self.v}.")

    @property
    def nvib(self):
        """The number of vibrational modes.

        :rtype: int
        """
        return len(self.v)
