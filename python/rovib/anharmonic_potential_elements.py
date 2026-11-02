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

"""Matrix elements of the anharmonic potential.

|Vanharm| is diagonal in :math:`J` and :math:`K`. Grouping the terms of the
cubic and quartic sums that are equal because the :math:`\\qop_i` commute
gives:

.. math::

   \\braket{\\vvec',J',K'\\mid\\Vanharm\\mid\\vvec,J,K} =&
      \\delta_{J',J}\\delta_{K',K}\\left[
      \\sum_{i\\le j\\le k}^{\\Nvib}
      \\frac{g_{ijk}}{\\prod_{n=1}^{\\Nvib} p_n!}
      \\prod_{n=1}^{\\Nvib}\\mathcal{Q}^{(p_n)}_{v'_n,v_n}\\right.\\\\
      &+\\left.\\sum_{i\\le j\\le k\\le l}^{\\Nvib}
      \\frac{h_{ijkl}}{\\prod_{n=1}^{\\Nvib} p_n!}
      \\prod_{n=1}^{\\Nvib}\\mathcal{Q}^{(p_n)}_{v'_n,v_n}\\right]

where :math:`p_n` is the number of times :math:`n` appears in the indices of
the force constant and
:math:`\\mathcal{Q}^{(p)}_{v'_n,v_n} = \\braket{v'_n\\mid\\qop_n^p\\mid v_n}`
(:math:`\\mathcal{Q}^{(0)}_{v'_n,v_n} = \\delta_{v'_n,v_n}`). See
:ref:`anharmonic_potential` and :ref:`vibrational_wavefunction` for the
derivation.
"""

import collections
import functools
import itertools
import math

from .rovibrational_basis_function import RovibrationalBasisFunction

#: Maps (p, v' - v) to :math:`\braket{v'\mid(a^\dagger + a)^p\mid v}` as a
#: function of v, for v' >= v. Elements with v' < v follow from the matrix
#: being symmetric.
_LADDER_POWERS = {
    (1, 1): lambda v: math.sqrt(v + 1),
    (2, 0): lambda v: 2 * v + 1,
    (2, 2): lambda v: math.sqrt((v + 1) * (v + 2)),
    (3, 1): lambda v: 3 * (v + 1) ** 1.5,
    (3, 3): lambda v: math.sqrt((v + 1) * (v + 2) * (v + 3)),
    (4, 0): lambda v: 6 * v**2 + 6 * v + 3,
    (4, 2): lambda v: (4 * v + 6) * math.sqrt((v + 1) * (v + 2)),
    (4, 4): lambda v: math.sqrt((v + 1) * (v + 2) * (v + 3) * (v + 4)),
}


def _single_mode_element(p, v_bra, v_ket, omega):
    """Computes :math:`\\mathcal{Q}^{(p)}_{v',v}` in atomic units."""
    if v_bra < v_ket:
        v_bra, v_ket = v_ket, v_bra
    ladder_power = _LADDER_POWERS.get((p, v_bra - v_ket))
    if ladder_power is None:
        return 0.0
    return (0.5 / omega) ** (p / 2) * ladder_power(v_ket)


# N.b. the force constants of a term are read-only, so its monomials can be
# cached.
@functools.cache
def _monomials(term):
    """Returns the distinct, non-zero terms of |Vanharm|.

    Each term is a pair. The first element is a tuple of (mode, power) pairs,
    one per mode in the term. The second element is the coefficient, i.e.,
    the force constant divided by the factorial of each power.
    """
    monomials = []
    for force_constants in (term.cubic, term.quartic):
        modes = range(term.nvib)
        order = force_constants.ndim
        for idx in itertools.combinations_with_replacement(modes, order):
            coefficient = float(force_constants[idx])
            if coefficient == 0.0:
                continue
            powers = collections.Counter(idx)
            coefficient /= math.prod(
                math.factorial(p) for p in powers.values()
            )
            monomials.append((tuple(powers.items()), coefficient))
    return tuple(monomials)


def anharmonic_matrix_element(bra, term, ket):
    """Computes a single matrix element of |Vanharm|.

    :param bra: The basis function in the bra.
    :type bra: RovibrationalBasisFunction
    :param term: The anharmonic potential.
    :type term: AnharmonicPotential
    :param ket: The basis function in the ket.
    :type ket: RovibrationalBasisFunction
    :return: The matrix element in hartree.
    :rtype: complex
    """
    if bra.J != ket.J or bra.K != ket.K:
        return 0j

    changed = {n for n, (vb, vk) in enumerate(zip(bra.v, ket.v)) if vb != vk}
    value = 0.0
    for powers, coefficient in _monomials(term):
        # Modes not in the monomial must have the same quantum number.
        if not changed.issubset(n for n, _ in powers):
            continue
        element = coefficient
        for n, p in powers:
            element *= _single_mode_element(
                p, bra.v[n], ket.v[n], term.frequencies[n]
            )
        value += element
    return complex(value)


def anharmonic_non_zero_indices(bra_space, term, ket_space):
    """Generates the bra-ket pairs with non-zero matrix elements of |Vanharm|.

    For a term with :math:`\\qop_n^{p_n}`, an element is non-zero only if
    :math:`v'_n - v_n` is one of :math:`-p_n, -p_n+2, \\ldots, p_n` and
    every mode not in the term has the same quantum number in the bra and
    ket. The bra and ket must also have the same :math:`J` and :math:`K`.

    :param bra_space: The space the bra basis functions come from.
    :type bra_space: RovibrationalAnsatz
    :param term: The anharmonic potential.
    :type term: AnharmonicPotential
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
        # Different terms can couple the same bra and ket, so the bras are
        # collected in a dict (an insertion-ordered set) to remove repeats.
        bras = {}
        for powers, _ in _monomials(term):
            shifts = [range(-p, p + 1, 2) for _, p in powers]
            for delta_v in itertools.product(*shifts):
                v = list(ket.v)
                for (n, _), delta in zip(powers, delta_v):
                    v[n] += delta
                if min(v) < 0:
                    continue
                bra = RovibrationalBasisFunction(tuple(v), ket.J, ket.K)
                if bra in bra_space:
                    bras[bra] = None
        for bra in bras:
            yield bra, ket
