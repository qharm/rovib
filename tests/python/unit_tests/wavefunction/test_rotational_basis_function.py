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

from rovib import RotationalBasisFunction


class TestRotationalBasisFunction(unittest.TestCase):
    def test_state(self):
        bf = RotationalBasisFunction(3, -2)
        self.assertEqual(bf.J, 3)
        self.assertEqual(bf.K, -2)

    def test_negative_J(self):
        with self.assertRaises(ValueError):
            RotationalBasisFunction(-1, 0)

    def test_K_out_of_range(self):
        with self.assertRaises(ValueError):
            RotationalBasisFunction(1, 2)
        with self.assertRaises(ValueError):
            RotationalBasisFunction(1, -2)

    def test_immutable(self):
        bf = RotationalBasisFunction(1, 0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            bf.J = 2

    def test_equality_and_hash(self):
        bf = RotationalBasisFunction(1, 1)
        same = RotationalBasisFunction(1, 1)
        self.assertEqual(bf, same)
        self.assertEqual(hash(bf), hash(same))
        self.assertNotEqual(bf, RotationalBasisFunction(1, 0))
