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

"""Matrix elements of the rigid rotor Hamiltonian.

|HRR| is diagonal in :math:`\\vvec` and :math:`J`, and couples :math:`K` only
to :math:`K` and :math:`K\\pm 2`:

.. math::

   \\braket{\\vvec',J',K'\\mid \\HRR\\mid\\vvec,J,K} =
      \\delta_{\\vvec',\\vvec}\\delta_{J',J}\\left(
      T^{(\\text{RR},0)}_{J,K',K} +
      T^{(\\text{RR},-2)}_{J,K',K} +
      T^{(\\text{RR},2)}_{J,K',K}\\right)

where:

.. math::

   T^{(\\text{RR},0)}_{J,K',K} \\equiv& \\delta_{K',K}\\left[
                                      \\frac{\\gamma_x+\\gamma_y}{2}
                                      \\left(J(J+1) - K^2\\right) +
                                      \\gamma_z K^2\\right]\\\\
   T^{(\\text{RR},-2)}_{J,K',K} \\equiv& \\delta_{K',K-2}
                                      \\frac{\\gamma_x-\\gamma_y}{4}
                                      C^-_{J,K}C^-_{J,K-1}\\\\
   T^{(\\text{RR},2)}_{J,K',K} \\equiv& \\delta_{K',K+2}
                                      \\frac{\\gamma_x-\\gamma_y}{4}
                                      C^+_{J,K}C^+_{J,K+1}

and :math:`C^\\pm_{J,K} = \\sqrt{J(J+1) - K(K\\pm1)}`. See
:ref:`rigid_rotor` for the derivation.
"""

import math

from ..wavefunction.rovibrational_basis_function import (
    RovibrationalBasisFunction,
)


def _ladder_coefficient(J, K, sign):
    """Computes :math:`C^\\pm_{J,K}`, with ``sign`` being +1 or -1."""
    return math.sqrt(J * (J + 1) - K * (K + sign))


def rigid_rotor_matrix_element(bra, term, ket):
    """Computes a single matrix element of |HRR|.

    :param bra: The basis function in the bra.
    :type bra: RovibrationalBasisFunction
    :param term: The rigid rotor Hamiltonian.
    :type term: RigidRotorHamiltonian
    :param ket: The basis function in the ket.
    :type ket: RovibrationalBasisFunction
    :return: The matrix element in hartree.
    :rtype: complex
    """
    if bra.v != ket.v or bra.J != ket.J:
        return 0j

    gamma_x, gamma_y, gamma_z = term.rotational_constants
    J, K = ket.J, ket.K
    delta_K = bra.K - K

    if delta_K == 0:
        return complex(
            0.5 * (gamma_x + gamma_y) * (J * (J + 1) - K**2)
            + gamma_z * K**2
        )
    if delta_K in (-2, 2):
        sign = delta_K // 2
        c = _ladder_coefficient(J, K, sign)
        c *= _ladder_coefficient(J, K + sign, sign)
        return complex(0.25 * (gamma_x - gamma_y) * c)
    return 0j


def rigid_rotor_non_zero_indices(bra_space, term, ket_space):
    """Generates the bra-ket pairs with non-zero matrix elements of |HRR|.

    An element is non-zero only if the bra and ket have the same vibrational
    quantum numbers and :math:`J`, and their :math:`K` values differ by 0 or
    2.

    :param bra_space: The space the bra basis functions come from.
    :type bra_space: RovibrationalAnsatz
    :param term: The rigid rotor Hamiltonian.
    :type term: RigidRotorHamiltonian
    :param ket_space: The space the ket basis functions come from.
    :type ket_space: RovibrationalAnsatz
    :return: A generator of (bra, ket) pairs of
        :class:`.RovibrationalBasisFunction` objects.
    :rtype: collections.abc.Iterator
    """
    for ket in ket_space:
        for delta_K in (-2, 0, 2):
            K = ket.K + delta_K
            if abs(K) > ket.J:
                continue
            bra = RovibrationalBasisFunction(ket.v, ket.J, K)
            if bra in bra_space:
                yield bra, ket
