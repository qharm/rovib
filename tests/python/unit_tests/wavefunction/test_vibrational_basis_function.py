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

from rovib import VibrationalBasisFunction


class TestVibrationalBasisFunction(unittest.TestCase):
    def test_state(self):
        bf = VibrationalBasisFunction((1, 0, 2))
        self.assertEqual(bf.v, (1, 0, 2))
        self.assertEqual(bf.nvib, 3)

    def test_v_is_converted_to_tuple(self):
        self.assertEqual(VibrationalBasisFunction([1, 0, 2]).v, (1, 0, 2))

    def test_negative_v(self):
        with self.assertRaises(ValueError):
            VibrationalBasisFunction((0, -1, 0))

    def test_immutable(self):
        bf = VibrationalBasisFunction((0, 0, 0))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            bf.v = (1, 0, 0)

    def test_equality_and_hash(self):
        bf = VibrationalBasisFunction((1, 0, 2))
        same = VibrationalBasisFunction([1, 0, 2])
        self.assertEqual(bf, same)
        self.assertEqual(hash(bf), hash(same))
        self.assertNotEqual(bf, VibrationalBasisFunction((1, 0, 1)))
