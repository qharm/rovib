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

"""Matrix elements of the harmonic oscillator Hamiltonian.

|HHO| is diagonal in the rovibrational basis:

.. math::

    \\braket{\\vvec',J',K'\\mid\\HHO\\mid\\vvec,J,K} =
       \\delta_{\\vvec',\\vvec}\\delta_{J',J}\\delta_{K',K}
       \\sum_{i=1}^{\\Nvib}\\omega_i\\left(v_i + \\frac{1}{2}\\right)

in atomic units.
"""

import numpy as np


def harmonic_matrix_element(bra, term, ket):
    """Computes a single matrix element of |HHO|.

    :param bra: The basis function in the bra.
    :type bra: RovibrationalBasisFunction
    :param term: The harmonic oscillator Hamiltonian.
    :type term: HarmonicHamiltonian
    :param ket: The basis function in the ket.
    :type ket: RovibrationalBasisFunction
    :return: The matrix element in hartree.
    :rtype: complex
    """
    if bra != ket:
        return 0j
    return complex(np.dot(term.frequencies, np.add(ket.v, 0.5)))


def harmonic_non_zero_indices(bra_space, term, ket_space):
    """Generates the bra-ket pairs with non-zero matrix elements of |HHO|.

    Since |HHO| is diagonal, an element is non-zero only if the bra and ket
    are the same basis function.

    :param bra_space: The space the bra basis functions come from.
    :type bra_space: RovibrationalAnsatz
    :param term: The harmonic oscillator Hamiltonian.
    :type term: HarmonicHamiltonian
    :param ket_space: The space the ket basis functions come from.
    :type ket_space: RovibrationalAnsatz
    :return: A generator of (bra, ket) pairs of
        :class:`.RovibrationalBasisFunction` objects.
    :rtype: collections.abc.Iterator
    :raises ValueError: If either space has a different number of
        vibrational modes than ``term``.
    """
    for space in (bra_space, ket_space):
        if space.nvib != term.nvib:
            raise ValueError(
                f"The Hamiltonian has {term.nvib} modes, but the space has "
                f"{space.nvib}."
            )
    for ket in ket_space:
        if ket in bra_space:
            yield ket, ket
