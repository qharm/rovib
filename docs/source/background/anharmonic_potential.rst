.. Copyright 2026 QHARM Team
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
.. http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

.. _anharmonic_potential:

####################
Anharmonic Potential
####################

The anharmonic potential operator is defined as:

.. math::

    \Vanharm \equiv& \frac{1}{3!}
                     \sum_{ijk=1}^{\Nvib} g_{ijk}\qop_i\qop_j\qop_k +\\
                   & \frac{1}{4!}
                     \sum_{ijkl=1}^{\Nvib} h_{ijkl}\qop_i\qop_j\qop_k\qop_l

where :math:`g_{ijk}` and :math:`h_{ijkl}` are the cubic and quartic force
constants.

For convenience we break the cubic sum up into a term in which no index is
repeated, a term in which one index is repeated, and a term in which three
indices are repeated. Since the :math:`\qop_i` commute and
:math:`g_{ijk}` is symmetric under permutation of its indices, this gives:

.. math::

    \frac{1}{3!}\sum_{ijk=1}^{\Nvib} g_{ijk}\qop_i\qop_j\qop_k =&
       \sum_{i<j<k}^{\Nvib} g_{ijk}\qop_i\qop_j\qop_k\\
    &+ \frac{1}{2}\sum_{i\neq j}^{\Nvib} g_{iij}\qop_i^2\qop_j\\
    &+ \frac{1}{3!}\sum_{i=1}^{\Nvib} g_{iii}\qop_i^3

Similarly, we break the quartic sum up into a term in which all four indices
are distinct, a term in which exactly two indices are equal, a term with two
distinct pairs of equal indices, a term in which exactly three indices are
equal, and a term in which all four indices are equal. This gives:

.. math::

    \frac{1}{4!}\sum_{ijkl=1}^{\Nvib} h_{ijkl}\qop_i\qop_j\qop_k\qop_l =&
       \sum_{i<j<k<l}^{\Nvib} h_{ijkl}\qop_i\qop_j\qop_k\qop_l\\
    &+ \frac{1}{2}\sum_{i=1}^{\Nvib}\sum_{\substack{j<k\\ j,k\neq i}}^{\Nvib}
       h_{iijk}\qop_i^2\qop_j\qop_k\\
    &+ \frac{1}{4}\sum_{i<j}^{\Nvib} h_{iijj}\qop_i^2\qop_j^2\\
    &+ \frac{1}{3!}\sum_{i\neq j}^{\Nvib} h_{iiij}\qop_i^3\qop_j\\
    &+ \frac{1}{4!}\sum_{i=1}^{\Nvib} h_{iiii}\qop_i^4

For what it's worth, the general form is:

.. math::

    \Vanharm =& \sum_{i\le j\le k}^{\Nvib}
                \frac{g_{ijk}}{\prod_{n=1}^{\Nvib} p_n!}
                \qop_i\qop_j\qop_k\\
              &+ \sum_{i\le j\le k\le l}^{\Nvib}
                \frac{h_{ijkl}}{\prod_{n=1}^{\Nvib} p_n!}
                \qop_i\qop_j\qop_k\qop_l

where :math:`p_n` is the power of :math:`\qop_n` in the term.

===============
Matrix Elements
===============

Because the operators on different modes commute, the matrix element
of each term factors into one matrix element per mode (see
:ref:`vibrational_wavefunction`):

.. math::

    \braket{\vvec'\mid\qop_i\qop_j\qop_k\mid\vvec} =&
       \prod_{n=1}^{\Nvib}\braket{v'_n\mid\qop_n^{p_n}\mid v_n}\\
    \braket{\vvec'\mid\qop_i\qop_j\qop_k\qop_l\mid\vvec} =&
       \prod_{n=1}^{\Nvib}\braket{v'_n\mid\qop_n^{p_n}\mid v_n}

For a mode that does not appear in the term, :math:`p_n=0` and its factor is
:math:`\delta_{v'_n,v_n}`. Collecting these factors into:

.. math::

    \delta^{(ij\ldots)}_{\vvec',\vvec} \equiv
       \prod_{n\notin\left\{i,j,\ldots\right\}}\delta_{v'_n,v_n}

and using the single-mode matrix elements
:math:`\mathcal{Q}^{(p)}_{v'_n,v_n}` from :ref:`vibrational_wavefunction`, the
matrix elements of :math:`\Vanharm` are:

.. math::

    \braket{\vvec'\mid \Vanharm\mid\vvec} =&
       \sum_{i<j<k}^{\Nvib} g_{ijk}T^{(111)}_{ijk}\\
    &+ \frac{1}{2}\sum_{i\neq j}^{\Nvib} g_{iij}T^{(210)}_{ij}\\
    &+ \frac{1}{3!}\sum_{i=1}^{\Nvib} g_{iii}T^{(300)}_{i}\\
    &+ \sum_{i<j<k<l}^{\Nvib} h_{ijkl}T^{(1111)}_{ijkl}\\
    &+ \frac{1}{2}\sum_{i=1}^{\Nvib}\sum_{\substack{j<k\\ j,k\neq i}}^{\Nvib}
       h_{iijk}T^{(211)}_{ijk}\\
    &+ \frac{1}{4}\sum_{i<j}^{\Nvib} h_{iijj}T^{(22)}_{ij}\\
    &+ \frac{1}{3!}\sum_{i\neq j}^{\Nvib} h_{iiij}T^{(31)}_{ij}\\
    &+ \frac{1}{4!}\sum_{i=1}^{\Nvib} h_{iiii}T^{(4)}_{i}

where the tensors :math:`T^{(pqr\ldots)}_{ijk\ldots}` are defined as:

.. math::

   T^{(111)}_{ijk} \equiv& \mathcal{Q}_{v'_i,v_i}\mathcal{Q}_{v'_j,v_j}
                           \mathcal{Q}_{v'_k,v_k}\delta^{(ijk)}_{\vvec',\vvec}\\
   T^{(210)}_{ij} \equiv& \mathcal{Q}^{(2)}_{v'_i,v_i}\mathcal{Q}_{v'_j,v_j}
                           \delta^{(ij)}_{\vvec',\vvec}\\
   T^{(300)}_{i} \equiv& \mathcal{Q}^{(3)}_{v'_i,v_i}
                         \delta^{(i)}_{\vvec',\vvec}\\
   T^{(1111)}_{ijkl} \equiv& \mathcal{Q}_{v'_i,v_i}\mathcal{Q}_{v'_j,v_j}
                             \mathcal{Q}_{v'_k,v_k}\mathcal{Q}_{v'_l,v_l}
                             \delta^{(ijkl)}_{\vvec',\vvec}\\
   T^{(211)}_{ijk} \equiv& \mathcal{Q}^{(2)}_{v'_i,v_i}\mathcal{Q}_{v'_j,v_j}
                            \mathcal{Q}_{v'_k,v_k}
                            \delta^{(ijk)}_{\vvec',\vvec}\\
   T^{(22)}_{ij} \equiv& \mathcal{Q}^{(2)}_{v'_i,v_i}
                          \mathcal{Q}^{(2)}_{v'_j,v_j}
                          \delta^{(ij)}_{\vvec',\vvec}\\
   T^{(31)}_{ij} \equiv& \mathcal{Q}^{(3)}_{v'_i,v_i}\mathcal{Q}_{v'_j,v_j}
                         \delta^{(ij)}_{\vvec',\vvec}\\
    T^{(4)}_{i} \equiv& \mathcal{Q}^{(4)}_{v'_i,v_i}\delta^{(i)}_{\vvec',\vvec}

In practice, the matrix elements of :math:`\Vanharm` are computed by
iterating over the :math:`\binom{\Nvib+2}{3}` cubic and
:math:`\binom{\Nvib+3}{4}` quartic terms. Then for each term, the non-zero
matrix elements are evaluated by using the selection rules for the relevant
single-mode matrix elements.

In the full ansantz, the matrix elements of :math:`\Vanharm` are diagonal in
:math:`J` and :math:`K`, so we can define a tensor:

.. math::

   \braket{\vvec', J', K' \mid \Vanharm\mid\vvec, J, K} =
    \delta_{J',J}\delta_{K',K}
      \braket{\vvec'\mid \Vanharm\mid\vvec}
