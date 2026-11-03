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

import math
import unittest

from rovib.operators.conjugate_momentum import ConjugateMomentum
from rovib.operators.normal_mode import NormalMode


class TestNormalMode(unittest.TestCase):
    def test_act(self):
        # Q|v> = (1 / (2 omega))^(1/2) (sqrt(v) |v-1> + sqrt(v+1) |v+1>).
        result = dict(NormalMode(0, 2.0).act(3))
        self.assertEqual(set(result), {2, 4})
        self.assertAlmostEqual(result[2], 0.5 * math.sqrt(3.0))
        self.assertAlmostEqual(result[4], 0.5 * 2.0)

    def test_act_ground_state(self):
        result = NormalMode(0, 2.0).act(0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 1)
        self.assertAlmostEqual(result[0][1], 0.5)

    def test_hashable(self):
        self.assertEqual(hash(NormalMode(1, 2.0)), hash(NormalMode(1, 2.0)))
        self.assertNotEqual(NormalMode(1, 2.0), ConjugateMomentum(1, 2.0))
