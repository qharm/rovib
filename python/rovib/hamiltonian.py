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

"""Defines the Hamiltonian class."""

from .hamiltonian_term import HamiltonianTerm


class Hamiltonian:
    """A Hamiltonian, i.e., a sum of Hamiltonian terms.

    The Hamiltonian only records which terms it contains. Its matrix is built
    by :func:`.build_hamiltonian_matrix`.

    :param terms: The terms in the Hamiltonian.
    :type terms: tuple of HamiltonianTerm
    :raises TypeError: If any element of ``terms`` is not a
        :class:`~rovib.hamiltonian_term.HamiltonianTerm`.
    """

    def __init__(self, terms=()):
        self._terms = tuple(terms)
        for term in self._terms:
            if not isinstance(term, HamiltonianTerm):
                raise TypeError(f"{term!r} is not a HamiltonianTerm.")

    @property
    def terms(self):
        """The terms in the Hamiltonian, in the order they were added.

        :rtype: tuple of HamiltonianTerm
        """
        return self._terms

    def __len__(self):
        return len(self._terms)

    def __iter__(self):
        return iter(self._terms)

    def __add__(self, other):
        if isinstance(other, Hamiltonian):
            return Hamiltonian(self._terms + other.terms)
        if isinstance(other, HamiltonianTerm):
            return Hamiltonian(self._terms + (other,))
        return NotImplemented
