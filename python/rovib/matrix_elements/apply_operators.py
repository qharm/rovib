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

"""Application of a product of operators to a single quantum number."""

import collections
import functools


# N.b. operators are immutable, so the result can be cached. This matters
# because it is needed for every ket.
@functools.cache
def apply_operators(ops, state):
    """Applies a product of operators to a state.

    Each operator must have a method ``act`` which takes a quantum number and
    returns the (quantum number, amplitude) pairs of the resulting state, e.g.,
    :class:`.NormalMode` or :class:`.JOperator`.

    :param ops: The operators, as a tuple. The rightmost operator acts first.
    :type ops: tuple
    :param state: The quantum number of the state being acted on.
    :type state: int
    :return: A dict mapping each quantum number :math:`s'` to
        :math:`\\braket{s'\\mid\\op{O}_1\\cdots\\op{O}_m\\mid s}`.
    :rtype: dict
    """
    amplitudes = {state: 1.0}
    for op in reversed(ops):
        result = collections.defaultdict(float)
        for state_in, amplitude in amplitudes.items():
            for state_out, element in op.act(state_in):
                result[state_out] += amplitude * element
        amplitudes = result
    return dict(amplitudes)
