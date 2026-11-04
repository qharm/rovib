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

"""Defines the HamiltonianTerm class."""


class HamiltonianTerm:
    """Base class for the terms of a Hamiltonian.

    Terms carry the state needed to evaluate their matrix elements, but
    otherwise serve as a domain-specific language for specifying which terms
    are in the Hamiltonian. Matrix elements are evaluated by separate
    functions (see :mod:`rovib.matrix_elements.hamiltonian_matrix`), which
    only rely on :meth:`terms` and :attr:`nvib`.

    Adding two terms, or a term and a
    :class:`~rovib.hamiltonian.Hamiltonian`, gives a
    :class:`~rovib.hamiltonian.Hamiltonian`.
    """

    @property
    def nvib(self):
        """The number of vibrational modes the term acts on.

        :return: The number of modes, or ``None`` if the term does not act on
            the vibrational modes.
        :rtype: int
        """
        return None

    def terms(self, kets):
        """The products of operators whose sum is the term.

        :param kets: The rotational part of the space the term acts on.
            Rotational operators depend on its :math:`J`.
        :type kets: RotationalAnsatz
        :return: One (vibrational operators, reducer, rotational operators)
            tuple per direct product in the term. The first two are the
            ``ops`` and ``reducer`` arguments of :func:`.vibrational_matrix`,
            and the last is the ``ops`` argument of
            :func:`.rotational_matrix`. Terms that are diagonal in a basis have
            no operators for that basis, which gives the identity.
        :rtype: tuple
        :raises TypeError: If the term does not define its matrix elements.
        """
        raise TypeError(
            f"Matrix elements of {type(self).__name__} are not implemented."
        )

    def __add__(self, other):
        # Imported here because hamiltonian.py imports this module.
        from .hamiltonian import Hamiltonian

        return Hamiltonian((self,)) + other
