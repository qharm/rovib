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

import dataclasses
import unittest

from rovib.operators.normal_mode import NormalMode
from rovib.operators.number_operator import NumberOperator


class TestNumberOperator(unittest.TestCase):
    def test_act(self):
        self.assertEqual(NumberOperator(1).act(3), [(3, 3.0)])

    def test_act_ground_state(self):
        self.assertEqual(NumberOperator(1).act(0), [(0, 0.0)])

    def test_immutable(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            NumberOperator(1).mode = 2

    def test_hashable(self):
        self.assertEqual(hash(NumberOperator(1)), hash(NumberOperator(1)))
        self.assertNotEqual(NumberOperator(1), NumberOperator(2))
        self.assertNotEqual(NumberOperator(1), NormalMode(1, 1.0))
