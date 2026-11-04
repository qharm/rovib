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

"""Defines the CoriolisOperator class."""

import numpy as np

from .conjugate_momentum import ConjugateMomentum
from .normal_mode import NormalMode


class CoriolisOperator:
    """The vibrational angular momentum |pivec|.

    Each component of the operator is

    .. math::

       \\piop_\\alpha = \\sum_{i,j=1}^{\\Nvib}
          \\zeta^\\alpha_{i,j}\\qop_i\\pop_j

    so :meth:`coefficient` returns a vector, with one element per component.
    The vectors are combined with those of the other factors of the operator
    by the reducer (see :func:`.vibrational_matrix`).

    Like the other operators made of sums of products of single-mode
    operators (see :ref:`classes`), :meth:`vibrational_operators` returns the
    products and :meth:`coefficient` returns the coefficient of one of them.

    :param coriolis_couplings: The Coriolis coupling constants |zetavec|,
        indexed by (:math:`i`, :math:`\\alpha`, :math:`j`). The shape is
        (``nvib``, 3, ``nvib``).
    :type coriolis_couplings: numpy.ndarray
    :param frequencies: The harmonic frequencies |omegavec| in hartree. The
        shape is (``nvib``,).
    :type frequencies: numpy.ndarray
    :raises ValueError: If ``coriolis_couplings`` is not of shape
        (``nvib``, 3, ``nvib``).
    """

    def __init__(self, coriolis_couplings, frequencies):
        coriolis_couplings = np.array(coriolis_couplings, dtype=float)
        nvib = len(frequencies)
        if coriolis_couplings.shape != (nvib, 3, nvib):
            raise ValueError(
                f"Expected Coriolis couplings of shape {(nvib, 3, nvib)}, "
                f"got {coriolis_couplings.shape}."
            )
        coriolis_couplings.setflags(write=False)
        q = [NormalMode(n, float(w)) for n, w in enumerate(frequencies)]
        p = [ConjugateMomentum(n, float(w)) for n, w in enumerate(frequencies)]
        # N.b. the products and coefficients are computed once, because
        # vibrational_operators is called for every ket.
        self._coefficients = {
            (q[i], p[j]): coriolis_couplings[i, :, j]
            for i in range(nvib)
            for j in range(nvib)
        }

    def vibrational_operators(self):
        """The products of a normal mode and a conjugate momentum.

        :return: One (:class:`.NormalMode`, :class:`.ConjugateMomentum`) tuple
            per pair of modes.
        :rtype: collections.abc.Iterable
        """
        return self._coefficients.keys()

    def coefficient(self, operators):
        """The coefficient of a product of a normal mode and a momentum.

        :param operators: A product returned by
            :meth:`vibrational_operators`.
        :type operators: tuple
        :return: The vector :math:`\\zeta^\\alpha_{i,j}` over
            :math:`\\alpha`. The shape is (3,).
        :rtype: numpy.ndarray
        """
        return self._coefficients[operators]
