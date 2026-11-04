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

from rovib.matrix_elements.apply_operators import apply_operators
from rovib.operators.normal_mode import NormalMode


class _Raise:
    """Multiplies by ``factor`` and raises the quantum number by one."""

    def __init__(self, factor):
        self.factor = factor

    def act(self, state):
        return [(state + 1, self.factor)]


class _Project:
    """Multiplies by the quantum number."""

    def act(self, state):
        return [(state, float(state))]


class TestApplyOperators(unittest.TestCase):
    def test_no_operators(self):
        self.assertEqual(apply_operators((), 3), {3: 1.0})

    def test_rightmost_acts_first(self):
        raises, project = _Raise(1.0), _Project()
        self.assertEqual(apply_operators((project, raises), 2), {3: 3.0})
        self.assertEqual(apply_operators((raises, project), 2), {3: 2.0})

    def test_paths_are_summed(self):
        q = NormalMode(0, 0.5)
        # Q^2|1> with Q = a + a^dagger, which has two paths back to |1>.
        result = apply_operators((q, q), 1)
        self.assertEqual(set(result), {1, 3})
        self.assertAlmostEqual(result[1], 3.0)
