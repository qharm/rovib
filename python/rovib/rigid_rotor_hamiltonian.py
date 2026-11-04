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

"""Defines the RigidRotorHamiltonian class."""

import numpy as np

from .hamiltonian_term import HamiltonianTerm
from .operators.j_operator import JOperator


class RigidRotorHamiltonian(HamiltonianTerm):
    """The rigid rotor Hamiltonian |HRR|.

    The rotational constants are stored as a read-only array.

    :param rotational_constants: The rotational constants |gammavec|, i.e.,
        :math:`(\\gamma_x, \\gamma_y, \\gamma_z)`, in hartree. :math:`K` is
        the projection of |jvec| on to the z-axis.
    :type rotational_constants: numpy.ndarray
    :raises ValueError: If ``rotational_constants`` does not have shape (3,),
        or if any rotational constant is not positive.
    """

    def __init__(self, rotational_constants):
        self._rotational_constants = np.array(
            rotational_constants, dtype=float
        )

        if self._rotational_constants.shape != (3,):
            raise ValueError(
                "Expected three rotational constants, got shape "
                f"{self._rotational_constants.shape}."
            )
        if np.any(self._rotational_constants <= 0.0):
            raise ValueError("Rotational constants must be positive.")

        self._rotational_constants.setflags(write=False)

    @property
    def rotational_constants(self):
        """The rotational constants |gammavec| in hartree.

        :rtype: numpy.ndarray
        """
        return self._rotational_constants

    def rotational_operators(self, ket):
        """The products of angular momentum operators in |HRR|.

        :param ket: The space the operators act on.
        :type ket: RotationalAnsatz
        :return: The factors of :math:`\\jop_\\alpha^2`, with one element
            for each element of :attr:`rotational_constants`, in the same
            order.
        :rtype: tuple of list of JOperator
        """
        return tuple(
            [JOperator(ket.J, projection)] * 2
            for projection in range(self._rotational_constants.size)
        )
