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


class TestConjugateMomentum(unittest.TestCase):
    def test_act(self):
        # P|v> = i (omega / 2)^(1/2) (sqrt(v+1) |v+1> - sqrt(v) |v-1>).
        result = dict(ConjugateMomentum(0, 2.0).act(3))
        self.assertEqual(set(result), {2, 4})
        self.assertAlmostEqual(result[2], -1j * math.sqrt(3.0))
        self.assertAlmostEqual(result[4], 2j)

    def test_act_ground_state(self):
        result = ConjugateMomentum(0, 2.0).act(0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 1)
        self.assertAlmostEqual(result[0][1], 1j)
