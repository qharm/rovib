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

.. _vibration_coriolis:

################################
Vibrational-Coriolis Hamiltonian
################################

The vibrational Coriolis Hamiltonian is:

.. math::

   \HvibCor = \frac{1}{2}\sum_{n=0}^{2}\pivec^T\muvec^{(n)}\pivec -
              \frac{\hbar^2}{8}\sum_{n=0}^{4}\tr(\muvec^{(n)})


Writing out the matrix products and inserting the Taylor expansion of |muvec|:

.. math::

   \HvibCor =& \frac{1}{2}\sum_{n=0}^{2}\sum_{\alpha,\beta=1}^{3}
      \piop_\alpha\muop^{(n)}_{\alpha\beta}\piop_\beta -
       \frac{\hbar^2}{8}\sum_{n=0}^{4}
       \sum_{\alpha=1}^{3}\muop^{(n)}_{\alpha\alpha}\\
     =& \frac{1}{2}\sum_{n=0}^{2}\sum_{\alpha,\beta=1}^{3}
      \sum_{i,j,k,\ell=1}^{\Nvib}\sum_{\midx_n}
      \zeta^\alpha_{i,j}\zeta^\beta_{k,\ell}
      \eta^{\midx_n}_{\alpha\beta}
      \qop_i \pop_j \qop_{\midx_n} \qop_k \pop_\ell -\\
      & \frac{\hbar^2}{8}\sum_{n=0}^{4}
       \sum_{\alpha=1}^{3}\sum_{\midx_n}
      \eta^{\midx_n}_{\alpha\alpha}\qop_{\midx_n}

where we have used the definition of :math:`\muvec^{(n)}` from
:ref:`watson_mu_tensor`. The operator ordering must be kept,
because :math:`\pop_j` does not commute with :math:`\qop_{\midx_n}\qop_k`.

===============
Matrix Elements
===============

|HvibCor| is purely vibrational, so it suffices to work out the matrix
elements in the vibrational basis. The matrix elements are:

.. math::

   \braket{\vvec'\mid \HvibCor\mid\vvec}=&
   \frac{1}{2}\sum_{n=0}^{2}\sum_{\alpha,\beta=1}^{3}
      \sum_{i,j,k,\ell=1}^{\Nvib}\sum_{\midx_n}
      \zeta^\alpha_{i,j}\zeta^\beta_{k,\ell}
      \eta^{\midx_n}_{\alpha\beta}
      \braket{\vvec'\mid \qop_i \pop_j \qop_{\midx_n}
      \qop_k \pop_\ell \mid\vvec} -\\
      & \frac{\hbar^2}{8}\sum_{n=0}^{4}
       \sum_{\alpha=1}^{3}\sum_{\midx_n}
      \eta^{\midx_n}_{\alpha\alpha}\braket{\vvec'\mid \qop_{\midx_n}\mid\vvec}


Like the anharmonic potential (see :ref:`anharmonic_potential`), in the
vibrational basis, the vibrational Coriolis Hamiltonian is very sparse. Thus
evaluation of
:math:`\braket{\vvec'\mid \qop_i \pop_j \qop_{\midx_n} \qop_k \pop_\ell \mid\vvec}`
and :math:`\braket{\vvec'\mid \qop_{\midx_n}\mid\vvec}` is done using the
same technique (but now also needs the single-mode matrix elements of
:math:`\pop_j`). The :math:`n`\ th Coriolis term is a product of :math:`n+4`
vibrational operators and the :math:`n`\ th Watson term is a product of
:math:`n`, which sets the range of :math:`\vvec'` each term couples to.

In the full basis, the vibrational Coriolis Hamiltonian is diagonal in :math:`J`
and :math:`K` and the matrix elements are:

.. math::

   \braket{\vvec',J',K'\mid \HvibCor\mid\vvec,J,K} =
      \delta_{J',J}\delta_{K',K}\braket{\vvec'\mid \HvibCor\mid\vvec}
