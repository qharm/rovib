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

.. _watson_mu_tensor:

################
Watson mu Tensor
################

The problem description notes that the Watson mu tensor, |muvec|, is expanded as
a fourth-order Taylor series about the minimum energy geometry,
:math:`\muvec \approx \sum_{n=0}^{4}\muvec^{(n)}`. This page works out the
details.

*******************
General Formulation
*******************

The :math:`n`-th term in the Taylor expansion of |muvec| is given by:

.. math::

   \muvec^{(n)} = c_n \Izmhalf \bvec^n \Izmhalf

with:

- :math:`c_n = (n+1)\left(-\frac{1}{2}\right)^n`
- :math:`\bvec = \sum_{k=1}^{\Nvib} \Izmhalf \avec_k \Izmhalf \qop_k`

and :math:`\avec_k` is a 3 by 3 matrix with elements:

.. math::

    a_{k,\alpha\beta} =
       2 \sum_{i=1}^{\Natom} \sum_{\gamma,\delta=1}^{3}
       \left(\delta_{\alpha\beta}\delta_{\gamma\delta} -
             \delta_{\alpha\delta}\delta_{\beta\gamma}\right)
       \sqrt{M_i}\,
       r_{i,\gamma}^0
       L_{i,k,\delta}

***************
0-th Order Term
***************

The 0-th order term is straightforward:

.. math::

   \muvec^{(0)} =& c_0 \Izmhalf \bvec^0 \Izmhalf\\
                 =& 1 \cdot \Izmhalf \Izmhalf\\
                 =& \left(\Izvec\right)^{-1}

or in components:

.. math::

   \muop^{(0)}_{\alpha\beta} = \frac{\delta_{\alpha\beta}}
                                    {I^0_{\alpha\alpha}}


where we used the fact that :math:`\Izvec` is diagonal.

***************
1-st Order Term
***************

.. math::

   \muvec^{(1)} =& c_1 \Izmhalf \bvec^1 \Izmhalf\\
                 =& - \Izmhalf \bvec \Izmhalf\\
                 =& - \sum_{k=1}^{\Nvib} \Izmhalf
                    \left(\Izmhalf \avec_k \Izmhalf\right) \Izmhalf
                    \qop_k\\
                 =& -\sum_{k=1}^{\Nvib}
                    \left(\Izvec\right)^{-1}\avec_k\left(\Izvec\right)^{-1}
                    \qop_k

or in components:

.. math::

   \muop^{(1)}_{\alpha\beta} =& -\sum_{k=1}^{\Nvib}
      \sum_{\gamma,\delta=1}^{3}
      \frac{\delta_{\alpha\gamma}}{I^0_{\alpha\alpha}}
      a_{k,\gamma\delta}
      \frac{\delta_{\delta\beta}}{I^0_{\beta\beta}}
      \qop_k\\
      =&-\sum_{k=1}^{\Nvib}
      \frac{1}{I^0_{\alpha\alpha}I^0_{\beta\beta}}
      a_{k,\alpha\beta}
      \qop_k\\
     =&-\sum_{k=1}^{\Nvib}
      \frac{\xi^k_{\alpha\beta}}{I^0_{\beta\beta}}
      \qop_k

where we have defined:

.. math::

   \xi^k_{\alpha\beta} \equiv \frac{a_{k,\alpha\beta}}{I^0_{\alpha\alpha}}

***************
2-nd Order Term
***************

.. math::

   \muvec^{(2)} =& c_2 \Izmhalf \bvec^2 \Izmhalf\\
                 =& \frac{3}{4} \Izmhalf \bvec^2 \Izmhalf\\
                 =& \frac{3}{4} \sum_{k,\ell=1}^{\Nvib}
                    \Izmhalf
                    \left(\Izmhalf \avec_k \Izmhalf\right)
                    \left(\Izmhalf \avec_\ell \Izmhalf\right)\\
                  & \Izmhalf
                    \qop_k\qop_\ell\\
                 =& \frac{3}{4} \sum_{k,\ell=1}^{\Nvib}
                    \left(\Izvec\right)^{-1}\avec_k
                    \left(\Izvec\right)^{-1}\avec_\ell
                    \left(\Izvec\right)^{-1}
                    \qop_k\qop_\ell

or in components:

.. math::

   \muop^{(2)}_{\alpha\beta} =& \frac{3}{4}\sum_{k,\ell=1}^{\Nvib}
      \sum_{\gamma,\delta,\epsilon,\zeta=1}^{3}
      \frac{\delta_{\alpha\gamma}}{I^0_{\alpha\alpha}}
      a_{k,\gamma\delta}
      \frac{\delta_{\delta\epsilon}}{I^0_{\delta\delta}}
      a_{\ell,\epsilon\zeta}
      \frac{\delta_{\zeta\beta}}{I^0_{\beta\beta}}
      \qop_k\qop_\ell\\
      =& \frac{3}{4}\sum_{k,\ell=1}^{\Nvib}
      \sum_{\delta=1}^{3}
      \frac{1}{I^0_{\alpha\alpha}}
      a_{k,\alpha\delta}
      \frac{1}{I^0_{\delta\delta}}
      a_{\ell,\delta\beta}
      \frac{1}{I^0_{\beta\beta}}
      \qop_k\qop_\ell\\
      =& \frac{3}{4}\sum_{k,\ell=1}^{\Nvib}
      \sum_{\delta=1}^{3}
      \frac{1}{I^0_{\alpha\alpha}I^0_{\delta\delta}I^0_{\beta\beta}}
      a_{k,\alpha\delta}
      a_{\ell,\delta\beta}
      \qop_k\qop_\ell\\
     =& \frac{3}{4}\sum_{k,\ell=1}^{\Nvib}
      \sum_{\delta=1}^{3}
      \frac{\xi^k_{\alpha\delta}\xi^\ell_{\delta\beta}}
           {I^0_{\beta\beta}}\qop_k\qop_\ell

***************
3-rd Order Term
***************

.. math::

   \muvec^{(3)} =& c_3 \Izmhalf \bvec^3 \Izmhalf\\
                 =& -\frac{1}{2} \Izmhalf \bvec^3 \Izmhalf\\
                 =& -\frac{1}{2} \sum_{k,\ell,m=1}^{\Nvib}
                    \Izmhalf
                    \left(\Izmhalf \avec_k \Izmhalf\right)
                    \left(\Izmhalf \avec_\ell \Izmhalf\right)\\
                  & \left(\Izmhalf \avec_m \Izmhalf\right)
                    \Izmhalf
                    \qop_k\qop_\ell\qop_m\\
                 =& -\frac{1}{2} \sum_{k,\ell,m=1}^{\Nvib}
                    \left(\Izvec\right)^{-1}\avec_k
                    \left(\Izvec\right)^{-1}\avec_\ell
                    \left(\Izvec\right)^{-1}\avec_m
                    \left(\Izvec\right)^{-1}
                    \qop_k\qop_\ell\qop_m

and in components:

.. math::

   \muop^{(3)}_{\alpha\beta} =& -\frac{1}{2}\sum_{k,\ell,m=1}^{\Nvib}
      \sum_{\gamma,\delta,\epsilon,\zeta,\theta,\iota=1}^{3}
      \frac{\delta_{\alpha\gamma}}{I^0_{\alpha\alpha}}
      a_{k,\gamma\delta}
      \frac{\delta_{\delta\epsilon}}{I^0_{\delta\delta}}
      a_{\ell,\epsilon\zeta}
      \frac{\delta_{\zeta\theta}}{I^0_{\zeta\zeta}}
      a_{m,\theta\iota}
      \frac{\delta_{\iota\beta}}{I^0_{\beta\beta}}
      \qop_k\qop_\ell\qop_m\\
      =& -\frac{1}{2}\sum_{k,\ell,m=1}^{\Nvib}
      \sum_{\delta,\zeta=1}^{3}
      \frac{1}{I^0_{\alpha\alpha}I^0_{\delta\delta}
               I^0_{\zeta\zeta}I^0_{\beta\beta}}
      a_{k,\alpha\delta}
      a_{\ell,\delta\zeta}
      a_{m,\zeta\beta}
      \qop_k\qop_\ell\qop_m\\
      =& -\frac{1}{2}\sum_{k,\ell,m=1}^{\Nvib}
      \sum_{\delta,\zeta=1}^{3}
      \frac{\xi^k_{\alpha\delta}\xi^\ell_{\delta\zeta}\xi^m_{\zeta\beta}}
           {I^0_{\beta\beta}}
      \qop_k\qop_\ell\qop_m

***************
4-th Order Term
***************

.. math::

   \muvec^{(4)} =& c_4 \Izmhalf \bvec^4 \Izmhalf\\
                 =& \frac{5}{16} \Izmhalf \bvec^4 \Izmhalf\\
                 =& \frac{5}{16} \sum_{k,\ell,m,p=1}^{\Nvib}
                    \Izmhalf
                    \left(\Izmhalf \avec_k \Izmhalf\right)
                    \left(\Izmhalf \avec_\ell \Izmhalf\right)\\
                  & \left(\Izmhalf \avec_m \Izmhalf\right)
                    \left(\Izmhalf \avec_p \Izmhalf\right)
                    \Izmhalf
                    \qop_k\qop_\ell\qop_m\qop_p\\
                 =& \frac{5}{16} \sum_{k,\ell,m,p=1}^{\Nvib}
                    \left(\Izvec\right)^{-1}\avec_k
                    \left(\Izvec\right)^{-1}\avec_\ell
                    \left(\Izvec\right)^{-1}\avec_m
                    \left(\Izvec\right)^{-1}\avec_p
                    \left(\Izvec\right)^{-1}
                    \qop_k\qop_\ell\qop_m\qop_p

and in components:

.. math::

   \muop^{(4)}_{\alpha\beta} =& \frac{5}{16}\sum_{k,\ell,m,p=1}^{\Nvib}
      \sum_{\gamma,\delta,\epsilon,\zeta,\theta,\iota,\kappa,\lambda=1}^{3}
      \frac{\delta_{\alpha\gamma}}{I^0_{\alpha\alpha}}
      a_{k,\gamma\delta}
      \frac{\delta_{\delta\epsilon}}{I^0_{\delta\delta}}
      a_{\ell,\epsilon\zeta}
      \frac{\delta_{\zeta\theta}}{I^0_{\zeta\zeta}}
      a_{m,\theta\iota}
      \frac{\delta_{\iota\kappa}}{I^0_{\iota\iota}}
      a_{p,\kappa\lambda}
      \frac{\delta_{\lambda\beta}}{I^0_{\beta\beta}}
      \qop_k\qop_\ell\qop_m\qop_p\\
      =& \frac{5}{16}\sum_{k,\ell,m,p=1}^{\Nvib}
      \sum_{\delta,\zeta,\iota=1}^{3}
      \frac{1}{I^0_{\alpha\alpha}I^0_{\delta\delta}I^0_{\zeta\zeta}
               I^0_{\iota\iota}I^0_{\beta\beta}}
      a_{k,\alpha\delta}
      a_{\ell,\delta\zeta}
      a_{m,\zeta\iota}
      a_{p,\iota\beta}
      \qop_k\qop_\ell\qop_m\qop_p\\
      =& -\frac{5}{8}\sum_{k,\ell,m,p=1}^{\Nvib}
      \sum_{\delta,\zeta,\iota=1}^{3}
      \frac{\xi^k_{\alpha\delta}\xi^\ell_{\delta\zeta}
            \xi^m_{\zeta\iota}\xi^p_{\iota\beta}}
           {I^0_{\beta\beta}}
      \qop_k\qop_\ell\qop_m\qop_p


***************
n-th Order Term
***************

For the :math:`n`\ th order term it is helpful to define a multi-index
:math:`\midx_n = (i_1,i_2,\ldots,i_n)` with :math:`i_j \in [1,\Nvib]`. We can
then define the tensor :math:`\tens{\eta}` as:

.. math::

   \eta^{\midx_n}_{\alpha\beta} \equiv
      c_n\sum_{\delta_1,\delta_2,\ldots,\delta_{n-1}=1}^{3}
      \frac{\xi^{i_1}_{\alpha\delta_1}\xi^{i_2}_{\delta_1\delta_2}
            \xi^{i_3}_{\delta_2\delta_3}\cdots
            \xi^{i_n}_{\delta_{n-1}\beta}}
           {I^0_{\beta\beta}}

where the :math:`\xi` on the right-side are our previously defined 1-st order
terms. Also defining
:math:`\qop_{\midx_n} \equiv \qop_{i_1}\qop_{i_2}\cdots\qop_{i_n}`, for
:math:`n \geq 1` we can write the :math:`n`\ th order term as:

.. math::

   \muop^{(n)}_{\alpha\beta} = \sum_{\midx_n}
      \eta^{\midx_n}_{\alpha\beta}\qop_{\midx_n}

For :math:`n=0` the multi-index is empty, so the sum has a single term. Taking
:math:`\qop_{\midx_0} \equiv 1` and

.. math::

   \eta^{\midx_0}_{\alpha\beta} \equiv \frac{\delta_{\alpha\beta}}{I^0_{\alpha\alpha}}

the same expression reproduces the 0-th order term derived above, so it holds
for all :math:`n \geq 0`.
