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

.. _rovibrational_coupling:

##################################
Rovibrational Coupling Hamiltonian
##################################

The rovibrational coupling Hamiltonian is defined as:

.. math::

   \Hrovib \equiv \frac{1}{2}\sum_{n=1}^{4}\jvec^T\muvec^{(n)}\jvec -
      \frac{1}{2}\sum_{n=0}^{3}\left(\jvec^T\muvec^{(n)}\pivec +
      \pivec^T\muvec^{(n)}\jvec\right)


Since |jvec| is purely rotational it commutes with |qvec| and |pvec|, and thus
with :math:`\muvec^{(n)}` and |pivec|. :math:`\muvec^{(n)}` is also symmetric
(the individual :math:`\eta^{\midx_n}_{\alpha\beta}` are not, but reversing the
multi-index transposes them and the normal coordinates commute). Relabeling
:math:`\alpha\leftrightarrow\beta` then gives:

.. math::

   \pivec^T\muvec^{(n)}\jvec =
      \sum_{\alpha,\beta=1}^{3}\piop_\alpha\muop^{(n)}_{\alpha\beta}\jop_\beta =
      \sum_{\alpha,\beta=1}^{3}\jop_\alpha\piop_\beta\muop^{(n)}_{\alpha\beta}

and therefore:

.. math::

   \Hrovib = \frac{1}{2}\sum_{n=1}^{4}\jvec^T\muvec^{(n)}\jvec -
      \frac{1}{2}\sum_{n=0}^{3}\sum_{\alpha,\beta=1}^{3}\jop_\alpha\left(
      \muop^{(n)}_{\alpha\beta}\piop_\beta +
      \piop_\beta\muop^{(n)}_{\alpha\beta}\right)

Writing out the matrix products and inserting the Taylor expansion of |muvec|:

.. math::

   \Hrovib =& \frac{1}{2}\sum_{\alpha,\beta=1}^{3}
      \jop_\alpha\jop_\beta
      \sum_{n=1}^{4}\sum_{\midx_n}
      \eta^{\midx_n}_{\alpha\beta}\qop_{\midx_n} -\\
      &\frac{1}{2}\sum_{\alpha,\beta=1}^{3}
      \jop_\alpha
      \sum_{n=0}^{3}\sum_{\midx_n}\eta^{\midx_n}_{\alpha\beta}
      \sum_{i,j} \zeta^\beta_{ij}\left(
         \qop_{\midx_n}\qop_{i}\pop_{j} +
         \qop_{i}\pop_{j}\qop_{\midx_n}\right)

where we have used the definition of :math:`\muvec^{(n)}` from
:ref:`watson_mu_tensor`.

***************
Matrix Elements
***************

.. math::

   \braket{\vvec',J',K'\mid \Hrovib\mid\vvec,J,K}=&
   \frac{1}{2}\sum_{\alpha,\beta=1}^{3}
      \braket{J',K'\mid\jop_\alpha\jop_\beta\mid J,K}
      T_{\alpha\beta}^{(\text{RC},0)}
       -\\
      &\frac{1}{2}\sum_{\alpha=1}^{3}
       \braket{J',K'\mid\jop_\alpha\mid J,K}
       T_{\alpha}^{(\text{RC},1)}\\
   T_{\alpha\beta}^{(\text{RC},0)}\equiv&
      \sum_{n=1}^{4}\sum_{\midx_n} \eta^{\midx_n}_{\alpha\beta}
      \braket{\vvec'\mid\qop_{\midx_n}\mid\vvec}\\
   T_{\alpha}^{(\text{RC},1)}\equiv&
   \sum_{\beta=1}^{3}\sum_{n=0}^{3}\sum_{\midx_n}\eta^{\midx_n}_{\alpha\beta}
       \sum_{i,j} \zeta^\beta_{ij} \left(
         \braket{\vvec'\mid\qop_{\midx_n}\qop_{i}\pop_{j}\mid\vvec} +\right.\\
      &\left. \braket{\vvec'\mid\qop_{i}\pop_{j}\qop_{\midx_n}\mid\vvec}\right)
