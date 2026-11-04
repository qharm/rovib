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

"""The Coriolis coupling constants of a molecule."""

import numpy as np


def coriolis_couplings(normal_modes):
    """Computes the Coriolis coupling constants |zetavec|.

    .. math::

       \\zeta_{i,j}^\\alpha = \\sum_{k=1}^{\\Natom}
          \\left(\\Lvec_{ki}\\times\\Lvec_{kj}\\right)_\\alpha

    :param normal_modes: The normal-mode matrix |Lvec|, indexed by (atom,
        axis, mode).
    :type normal_modes: numpy.ndarray
    :return: The (dimensionless) Coriolis coupling constants, indexed by
        (:math:`i`, :math:`\\alpha`, :math:`j`). The shape is (number of
        modes, 3, number of modes).
    :rtype: numpy.ndarray
    """
    L = np.asarray(normal_modes, dtype=float)
    per_atom = np.cross(L[:, :, :, np.newaxis], L[:, :, np.newaxis, :], axis=1)
    return per_atom.sum(axis=0).transpose(1, 0, 2)
