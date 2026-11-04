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

import unittest

from rovib import Hamiltonian, HamiltonianTerm, RotationalAnsatz


class TestHamiltonianTerm(unittest.TestCase):
    def test_nvib(self):
        self.assertIsNone(HamiltonianTerm().nvib)

    def test_terms_not_implemented(self):
        with self.assertRaises(TypeError):
            HamiltonianTerm().terms(RotationalAnsatz(0))

    def test_add_term(self):
        t1 = HamiltonianTerm()
        t2 = HamiltonianTerm()
        h = t1 + t2
        self.assertIsInstance(h, Hamiltonian)
        self.assertEqual(h.terms, (t1, t2))

    def test_add_hamiltonian(self):
        t1 = HamiltonianTerm()
        t2 = HamiltonianTerm()
        t3 = HamiltonianTerm()
        h = t1 + Hamiltonian((t2, t3))
        self.assertEqual(h.terms, (t1, t2, t3))

    def test_add_unsupported(self):
        with self.assertRaises(TypeError):
            HamiltonianTerm() + 1
