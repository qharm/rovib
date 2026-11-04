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

"""The inertia tensor, its derivatives, and the rotational constants."""

import numpy as np


def inertia_tensor(molecule):
    """Computes the inertia tensor of a molecule about the origin.

    .. math::

       I_{\\alpha\\beta} = \\sum_{i=1}^{\\Natom} M_{ii}
          \\left[\\left(\\vec{r}_i\\cdot\\vec{r}_i\\right)
          \\delta_{\\alpha,\\beta} - r_{i,\\alpha}r_{i,\\beta}\\right]

    For the inertia tensor about the center of mass, the center of mass of
    ``molecule`` must be at the origin.

    :param molecule: The molecule.
    :type molecule: Molecule
    :return: The inertia tensor in electron masses times bohr squared. The
        shape is (3, 3).
    :rtype: numpy.ndarray
    """
    r = molecule.coords
    m = molecule.masses
    r2 = np.einsum("n,ni,ni->", m, r, r)
    return r2 * np.eye(3) - np.einsum("n,ni,nj->ij", m, r, r)


def inertia_derivatives(molecule, normal_modes):
    """Computes the derivatives |avec| of the inertia tensor of a molecule.

    .. math::

       a_{k,\\alpha\\beta} =
          2 \\sum_{i=1}^{\\Natom} \\sum_{\\gamma,\\delta=1}^{3}
          \\left(\\delta_{\\alpha\\beta}\\delta_{\\gamma\\delta} -
                \\delta_{\\alpha\\delta}\\delta_{\\beta\\gamma}\\right)
          \\sqrt{M_i}\\, r_{i,\\gamma}^0 L_{i,k,\\delta}

    :math:`\\avec_k` is only symmetric if |Lvec| satisfies the Eckart
    conditions at the geometry of ``molecule``. No attempt is made to
    symmetrize the result, so normal modes with limited precision give an
    :math:`\\avec_k` that is only symmetric to that precision.

    :param molecule: The molecule, at its minimum energy geometry, with its
        center of mass at the origin and its axes along the principal axes.
    :type molecule: Molecule
    :param normal_modes: The normal-mode matrix |Lvec|, indexed by (atom,
        axis, mode).
    :type normal_modes: numpy.ndarray
    :return: The derivatives in atomic units, indexed by (:math:`k`,
        :math:`\\alpha`, :math:`\\beta`). The shape is (number of modes, 3,
        3).
    :rtype: numpy.ndarray
    """
    r = molecule.coords
    L = np.asarray(normal_modes, dtype=float)
    weights = np.sqrt(molecule.masses)
    trace = np.einsum("n,ni,nik->k", weights, r, L)
    outer = np.einsum("n,nak,nb->kab", weights, L, r)
    return 2.0 * (np.einsum("k,ab->kab", trace, np.eye(3)) - outer)


def rotational_constants(molecule, ordering=(1, 3, 2)):
    """Computes the rotational constants |gammavec| of a molecule.

    The rotational constants are
    :math:`\\gamma_\\alpha = \\frac{\\hbar^2}{2I^0_{\\alpha\\alpha}}`, where
    the :math:`I^0_{\\alpha\\alpha}` are the principal moments of inertia,
    i.e., the eigenvalues of the inertia tensor |Izvec| at the minimum energy
    geometry.

    Diagonalizing |Izvec| does not say which principal axis is x, y, or z;
    ``ordering`` does. ``ordering[i]`` is the rank (1 being the smallest) of
    the rotational constant assigned to the i-th axis. The default, (1, 3,
    2), is the convention of doi:10.1063/5.0333947: x has the smallest
    rotational constant, y the largest, and z the middle one. The
    axes used for other quantities (e.g., the normal modes) must follow the
    same convention.

    :param molecule: The molecule, at its minimum energy geometry, with its
        center of mass at the origin.
    :type molecule: Molecule
    :param ordering: A permutation of (1, 2, 3) giving the rank of the
        rotational constant of each axis. A string such as ``"132"`` is also
        accepted.
    :type ordering: tuple of int
    :return: The rotational constants :math:`(\\gamma_x, \\gamma_y,
        \\gamma_z)` in hartree.
    :rtype: numpy.ndarray
    :raises ValueError: If ``ordering`` is not a permutation of (1, 2, 3), or
        if a principal moment of inertia is zero (i.e., the molecule is
        linear).
    """
    ranks = tuple(int(rank) for rank in ordering)
    if sorted(ranks) != [1, 2, 3]:
        raise ValueError(
            f"ordering must be a permutation of (1, 2, 3), got {ordering!r}."
        )

    # N.b. eigvalsh returns the moments in ascending order, so the
    # rotational constants are in descending order.
    moments = np.linalg.eigvalsh(inertia_tensor(molecule))
    # The smallest moment of a linear molecule is zero up to round-off.
    if moments[0] <= 1.0e-10 * moments[-1]:
        raise ValueError("The principal moments of inertia must be positive.")
    ascending = 1.0 / (2.0 * moments[::-1])
    return ascending[[rank - 1 for rank in ranks]]
