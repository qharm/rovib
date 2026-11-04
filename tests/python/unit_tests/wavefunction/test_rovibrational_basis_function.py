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

from rovib import (
    RotationalBasisFunction,
    RovibrationalBasisFunction,
    VibrationalBasisFunction,
)


class TestRovibrationalBasisFunction(unittest.TestCase):
    def test_state(self):
        bf = RovibrationalBasisFunction((1, 0, 2), 3, -2)
        self.assertEqual(bf.v, (1, 0, 2))
        self.assertEqual(bf.J, 3)
        self.assertEqual(bf.K, -2)
        self.assertEqual(bf.nvib, 3)
        self.assertEqual(bf.vibrational, VibrationalBasisFunction((1, 0, 2)))
        self.assertEqual(bf.rotational, RotationalBasisFunction(3, -2))

    def test_v_is_converted_to_tuple(self):
        bf = RovibrationalBasisFunction([1, 0, 2], 0, 0)
        self.assertEqual(bf.v, (1, 0, 2))

    def test_negative_v(self):
        with self.assertRaises(ValueError):
            RovibrationalBasisFunction((0, -1, 0), 0, 0)

    def test_negative_J(self):
        with self.assertRaises(ValueError):
            RovibrationalBasisFunction((0, 0, 0), -1, 0)

    def test_K_out_of_range(self):
        with self.assertRaises(ValueError):
            RovibrationalBasisFunction((0, 0, 0), 1, 2)
        with self.assertRaises(ValueError):
            RovibrationalBasisFunction((0, 0, 0), 1, -2)

    def test_immutable(self):
        bf = RovibrationalBasisFunction((0, 0, 0), 1, 0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            bf.J = 2

    def test_equality_and_hash(self):
        bf = RovibrationalBasisFunction((1, 0, 2), 1, 1)
        same = RovibrationalBasisFunction([1, 0, 2], 1, 1)
        self.assertEqual(bf, same)
        self.assertEqual(hash(bf), hash(same))
        self.assertNotEqual(bf, RovibrationalBasisFunction((1, 0, 2), 1, 0))
        self.assertEqual(len({bf, same}), 1)
