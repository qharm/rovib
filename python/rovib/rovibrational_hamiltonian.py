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

"""Defines the RovibrationalHamiltonian class."""

from .coriolis_hamiltonian_term import CoriolisHamiltonianTerm
from .operators.j_operator import JOperator

#: The orders of |muvec| in the :math:`\\jvec^T\\muvec\\jvec` part of |Hrovib|.
_CENTRIFUGAL_ORDERS = range(1, 5)

#: The orders of |muvec| in the Coriolis part of |Hrovib|.
_CORIOLIS_ORDERS = range(0, 4)


def _centrifugal_coefficient(alpha, beta):
    """Makes the reducer for :math:`\\frac{1}{2}\\jop_\\alpha\\jop_\\beta
    \\muop^{(n)}_{\\alpha\\beta}`.

    See :class:`RovibrationalHamiltonian` for why only the symmetric part of
    :math:`\\eta` is used.
    """

    def reducer(coefficients):
        eta = coefficients[0]
        return 0.25 * float(eta[alpha, beta] + eta[beta, alpha])

    return reducer


def _mu_pi_coefficient(alpha):
    """Makes the reducer for :math:`-\\frac{1}{2}\\jop_\\alpha\\sum_\\beta
    \\muop^{(n)}_{\\alpha\\beta}\\piop_\\beta`."""

    def reducer(coefficients):
        eta, zeta = coefficients
        return -0.5 * float(eta[alpha] @ zeta)

    return reducer


def _pi_mu_coefficient(alpha):
    """Makes the reducer for :math:`-\\frac{1}{2}\\jop_\\alpha\\sum_\\beta
    \\piop_\\beta\\muop^{(n)}_{\\alpha\\beta}`."""

    def reducer(coefficients):
        zeta, eta = coefficients
        return -0.5 * float(eta[alpha] @ zeta)

    return reducer


class RovibrationalHamiltonian(CoriolisHamiltonianTerm):
    """The rovibrational coupling Hamiltonian |Hrovib|.

    In atomic units,

    .. math::

       \\Hrovib = \\frac{1}{2}\\sum_{n=1}^{4}\\jvec^T\\muvec^{(n)}\\jvec -
          \\frac{1}{2}\\sum_{n=0}^{3}\\sum_{\\alpha,\\beta=1}^{3}\\jop_\\alpha
          \\left(\\muop^{(n)}_{\\alpha\\beta}\\piop_\\beta +
          \\piop_\\beta\\muop^{(n)}_{\\alpha\\beta}\\right)

    :math:`\\muvec^{(n)}` is symmetric only if |avec| is, and normal modes
    with limited precision give an |avec| that is only symmetric to that
    precision. Since :math:`\\jop_\\alpha\\jop_\\beta` is not Hermitian, the
    first term is only Hermitian if :math:`\\muvec^{(n)}` is symmetric, so its
    symmetric part is what is used. The Coriolis term is Hermitian for any
    |avec|, because it keeps both orderings of |pivec| and
    :math:`\\muvec^{(n)}`. See :ref:`rovibrational_coupling` for the matrix
    elements.

    :param frequencies: The harmonic frequencies |omegavec| in hartree. The
        shape is (number of vibrational modes,).
    :type frequencies: numpy.ndarray
    :param coriolis_couplings: The Coriolis coupling constants |zetavec|,
        indexed by (:math:`i`, :math:`\\alpha`, :math:`j`). The shape is
        (``nvib``, 3, ``nvib``).
    :type coriolis_couplings: numpy.ndarray
    :param inertia_derivatives: The derivatives of the inertia tensor |avec|
        in atomic units, indexed by (:math:`k`, :math:`\\alpha`,
        :math:`\\beta`). The shape is (``nvib``, 3, 3).
    :type inertia_derivatives: numpy.ndarray
    :param rotational_constants: The rotational constants |gammavec| in
        hartree. These must be in the same axes as ``coriolis_couplings`` and
        ``inertia_derivatives``.
    :type rotational_constants: numpy.ndarray
    :raises ValueError: If ``frequencies`` is not a non-empty vector, if any
        frequency or rotational constant is not positive, if any input has the
        wrong shape, or if ``coriolis_couplings`` is not antisymmetric in
        :math:`i` and :math:`j`.
    """

    def __init__(
        self,
        frequencies,
        coriolis_couplings,
        inertia_derivatives,
        rotational_constants,
    ):
        super().__init__(
            frequencies,
            coriolis_couplings,
            inertia_derivatives,
            rotational_constants,
        )
        pi, mu = self._pi, self._mu
        axes = range(3)
        # N.b. the inputs are read-only, so the vibrational part of each
        # product, including the reducers vibrational_matrix caches on, only
        # needs to be built once. The rotational part depends on J.
        self._products = tuple(
            ([mu[n]], _centrifugal_coefficient(alpha, beta), (alpha, beta))
            for n in _CENTRIFUGAL_ORDERS
            for alpha in axes
            for beta in axes
        ) + tuple(
            product
            for n in _CORIOLIS_ORDERS
            for alpha in axes
            for product in (
                ([mu[n], pi], _mu_pi_coefficient(alpha), (alpha,)),
                ([pi, mu[n]], _pi_mu_coefficient(alpha), (alpha,)),
            )
        )

    def terms(self, kets):
        """The centrifugal distortion and Coriolis parts of |Hrovib|.

        :param kets: The rotational part of the space the term acts on.
        :type kets: RotationalAnsatz
        :return: The products of :math:`\\jop_\\alpha\\jop_\\beta` for
            :math:`n = 1, \\ldots, 4`, followed by those of
            :math:`\\jop_\\alpha` for :math:`n = 0, \\ldots, 3`, in the form
            described by :meth:`.HamiltonianTerm.terms`.
        :rtype: tuple
        """
        return tuple(
            (
                vibrational,
                reducer,
                [JOperator(kets.J, axis) for axis in axes],
            )
            for vibrational, reducer, axes in self._products
        )
