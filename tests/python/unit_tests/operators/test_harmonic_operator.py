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

import numpy as np

from rovib.operators.harmonic_operator import HarmonicOperator
from rovib.operators.number_operator import NumberOperator


class TestHarmonicOperator(unittest.TestCase):
    def setUp(self):
        self.operator = HarmonicOperator([1.0, 2.0, 3.0])

    def test_vibrational_operators(self):
        self.assertEqual(
            set(self.operator.vibrational_operators()),
            {
                (),
                (NumberOperator(0),),
                (NumberOperator(1),),
                (NumberOperator(2),),
            },
        )

    def test_coefficient(self):
        self.assertEqual(self.operator.coefficient((NumberOperator(1),)), 2.0)

    def test_zero_point_energy(self):
        self.assertEqual(self.operator.coefficient(()), 3.0)

    def test_bad_shape(self):
        with self.assertRaises(ValueError):
            HarmonicOperator(np.ones((3, 1)))
