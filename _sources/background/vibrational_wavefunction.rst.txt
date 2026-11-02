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

.. _vibrational_wavefunction:

########################
Vibrational Wavefunction
########################

The point of this page is to derive relationships involving the vibrational
component of the rovibrational wavefunction, :math:`\ket{\vvec}`. Here,
:math:`\vvec` is a vector of vibrational quantum numbers, one for each of the
:math:`\Nvib` vibrational modes of the molecule.

Matrix elements involving the vibrational wavefunciton are derived using second
quantization. In mass-weighted normal coordinates:

.. math::

    \qop_n =& \sqrt{\frac{\hbar}{2\omega_n}}\left(\aop^\dagger_n +
              \aop_n\right)\\
    \pop_n =& i\sqrt{\frac{\hbar\omega_n}{2}}\left(\aop^\dagger_n -
              \aop_n\right)

where :math:`\aop^\dagger_n` and :math:`\aop_n` are the creation and
annihilation
operators for the :math:`n`-th vibrational mode, and :math:`\omega_n` is the
frequency of that mode. The action of these operators on the vibrational
eigenstates are given by:

.. math::

    \aop^\dagger_n\ket{v_1,\ldots,v_n,\ldots,v_{\Nvib}} =&
       \sqrt{v_n+1}\ket{v_1,\ldots,v_n+1,\ldots,v_{\Nvib}}\\
    \aop_n\ket{v_1,\ldots,v_n,\ldots,v_{\Nvib}} =&
       \sqrt{v_n}\ket{v_1,\ldots,v_n-1,\ldots,v_{\Nvib}}

Thus the action of :math:`\qop_n` and :math:`\pop_n` on the vibrational
eigenstates are:

.. math::

    \qop_n\ket{v_1,\ldots,v_n,\ldots,v_{\Nvib}} =&
       \sqrt{\frac{\hbar}{2\omega_n}}\left[
       \sqrt{v_n+1}\ket{v_1,\ldots,v_n+1,\ldots,v_{\Nvib}} +\right.\\
       &\left.\sqrt{v_n}\ket{v_1,\ldots,v_n-1,\ldots,v_{\Nvib}}\right]\\
    \pop_n\ket{v_1,\ldots,v_n,\ldots,v_{\Nvib}} =&
       i\sqrt{\frac{\hbar\omega_n}{2}}\left[
       \sqrt{v_n+1}\ket{v_1,\ldots,v_n+1,\ldots,v_{\Nvib}} -\right.\\
       &\left.\sqrt{v_n}\ket{v_1,\ldots,v_n-1,\ldots,v_{\Nvib}}\right]


Note that :math:`\qop_n` and :math:`\pop_n` obey the commutation relation:

.. math::

   \left[\qop_n,\pop_m\right] = i\hbar\delta_{nm}

Thus operators on different modes commute.

***************
Matrix Elements
***************

We will have to evaluate matrix elements of the form
:math:`\braket{\vvec'\mid \op{O}\mid\vvec}` where :math:`\op{O}` is a
product of :math:`\qop` and :math:`\pop` operators. Using the commutation
relation for :math:`n\neq m`, the product of :math:`\qop` and :math:`\pop`
operators factors into one matrix element per mode:

.. math::


   \braket{\vvec'\mid \op{O}_1 \op{O}_2\cdots \op{O}_p\mid\vvec} =
      \prod_{n=1}^{\Nvib}\braket{v'_n\mid \op{O}^{(n)}\mid v_n}

where :math:`\op{O}^{(n)}` is the product, in the original order, of the
operators that act on mode :math:`n` (the identity if there are none) and we
have used the fact that only :math:`\op{O}^{(n)}` acts on mode :math:`n`.

In general, each :math:`\op{O}^{(n)}` will be made up of products containing
:math:`\qop_n` and :math:`\pop_n` operators. Assuming :math:`\op{O}^{(n)}` is a
product of :math:`p` operators, we can invoke the resolution of the identity
:math:`p-1` times to factor the matrix element into a product of single-mode
matrix elements:

.. math::

   \braket{v'_n\mid \op{O}^{(n)}\mid v_n} =
      \sum_{v^{(2)}_n\ldots v^{(p)}_n}
      \braket{v'_n\mid \op{O}^{(n)}_1\mid v^{(2)}_n}
      \braket{v^{(2)}_n\mid \op{O}^{(n)}_2 \mid v^{(3)}_n}\cdots
      \braket{v^{(p)}_n\mid \op{O}^{(n)}_{p}\mid v_n}

where :math:`\op{O}^{(n)}_i` is the :math:`i`-th operator in the product
:math:`\op{O}^{(n)}` and we have labeled the intermediate states with
superscripts (as opposed to primes) for convenience. The single
mode matrix elements of :math:`\qop_n` and :math:`\pop_n` are:

.. math::

    \mathcal{Q}_{v'_n,v_n} \equiv \braket{v'_n \mid\qop_n\mid v_n} =&
       \sqrt{\frac{\hbar}{2\omega_n}}\left[\sqrt{v_n+1}\delta_{v'_n,v_n+1} +
       \sqrt{v_n}\delta_{v'_n,v_n-1}\right]\\
    \mathcal{P}_{v'_n,v_n} \equiv \braket{v'_n \mid\pop_n\mid v_n} =&
       i\sqrt{\frac{\hbar\omega_n}{2}}\left[\sqrt{v_n+1}\delta_{v'_n,v_n+1} -
       \sqrt{v_n}\delta_{v'_n,v_n-1}\right]

The matrices :math:`\mathcal{Q}` and :math:`\mathcal{P}` are extremely sparse,
and the matrix multiplication implied by an equation like:

.. math::

   \braket{v'_n\mid \op{Q}_n^2\mid v_n} =
      \sum_{v^{(2)}_n}\mathcal{Q}_{v'_n,v^{(2)}_n}\mathcal{Q}_{v^{(2)}_n,v_n}

is very inefficient. Instead we work out explicit formulas for the needed
powers

Squared
=======

.. math::

   \mathcal{Q}^{(2)}_{v'_n,v_n}\equiv&
   \braket{v'_n\mid \op{Q}_n^2\mid v_n} =
      \frac{\hbar}{2\omega_n}\sum_{v^{(2)}_n}
       \left[\sqrt{v^{(2)}_n+1}\delta_{v'_n,v^{(2)}_n+1} +
       \sqrt{v^{(2)}_n}\delta_{v'_n,v^{(2)}_n-1}\right]\\
       &
       \left[\sqrt{v_n+1}\delta_{v^{(2)}_n,v_n+1} +
       \sqrt{v_n}\delta_{v^{(2)}_n,v_n-1}\right]

The four cross terms:

.. math::

    \sum_{v^{(2)}_n}\sqrt{v^{(2)}_n+1}\delta_{v'_n,v^{(2)}_n+1}
                    \sqrt{v_n+1}\delta_{v^{(2)}_n,v_n+1} =&
    \sqrt{(v_n+2)(v_n+1)}\delta_{v'_n-2,v_n}\\
    \sum_{v^{(2)}_n}\sqrt{v^{(2)}_n+1}\delta_{v'_n,v^{(2)}_n+1}
                    \sqrt{v_n}\delta_{v^{(2)}_n,v_n-1} =&
    v_n\delta_{v'_n,v_n}\\
    \sum_{v^{(2)}_n}\sqrt{v^{(2)}_n}\delta_{v'_n,v^{(2)}_n-1}
       \sqrt{v_n+1}\delta_{v^{(2)}_n,v_n+1}=&
       \left(v_n+1\right)\delta_{v'_n,v_n}\\
    \sum_{v^{(2)}_n}\sqrt{v^{(2)}_n}\delta_{v'_n,v^{(2)}_n-1}
        \sqrt{v_n}\delta_{v^{(2)}_n,v_n-1}=&
        \sqrt{\left(v_n-1\right)\left(v_n\right)}\delta_{v'_n+2,v_n}

and the result:

.. math::

   \mathcal{Q}^{(2)}_{v'_n,v_n} =& \frac{\hbar}{2\omega_n}\left[
    \sqrt{(v_n+2)(v_n+1)}\delta_{v'_n-2,v_n}+
    \left(2v_n+1\right)\delta_{v'_n,v_n}+\right.\\
    &\left.
    \sqrt{\left(v_n-1\right)\left(v_n\right)}\delta_{v'_n+2,v_n}
    \right]

Cubed
=====

Inserting the resolution of the identity between :math:`\op{Q}_n^2` and
:math:`\op{Q}_n`, and using the squared result:

.. math::

   \mathcal{Q}^{(3)}_{v'_n,v_n}\equiv&
   \braket{v'_n\mid \op{Q}_n^3\mid v_n} =
      \sum_{v^{(2)}_n}\braket{v'_n\mid \op{Q}_n^2\mid v^{(2)}_n}
      \braket{v^{(2)}_n\mid \op{Q}_n\mid v_n}\\
      =& \left(\frac{\hbar}{2\omega_n}\right)^{3/2}\sum_{v^{(2)}_n}
       \left[\sqrt{(v^{(2)}_n+2)(v^{(2)}_n+1)}\delta_{v'_n-2,v^{(2)}_n}+
       \left(2v^{(2)}_n+1\right)\delta_{v'_n,v^{(2)}_n}+\right.\\
       &\left.\sqrt{\left(v^{(2)}_n-1\right)\left(v^{(2)}_n\right)}
       \delta_{v'_n+2,v^{(2)}_n}\right]
       \left[\sqrt{v_n+1}\delta_{v^{(2)}_n,v_n+1} +
       \sqrt{v_n}\delta_{v^{(2)}_n,v_n-1}\right]

The six cross terms:

.. math::

    \sum_{v^{(2)}_n}\sqrt{(v^{(2)}_n+2)(v^{(2)}_n+1)}\delta_{v'_n-2,v^{(2)}_n}
       \sqrt{v_n+1}\delta_{v^{(2)}_n,v_n+1} =&
       \sqrt{(v_n+3)(v_n+2)(v_n+1)}\delta_{v'_n-3,v_n}\\
    \sum_{v^{(2)}_n}\sqrt{(v^{(2)}_n+2)(v^{(2)}_n+1)}\delta_{v'_n-2,v^{(2)}_n}
       \sqrt{v_n}\delta_{v^{(2)}_n,v_n-1} =&
       v_n\sqrt{v_n+1}\delta_{v'_n-1,v_n}\\
    \sum_{v^{(2)}_n}\left(2v^{(2)}_n+1\right)\delta_{v'_n,v^{(2)}_n}
       \sqrt{v_n+1}\delta_{v^{(2)}_n,v_n+1} =&
       \left(2v_n+3\right)\sqrt{v_n+1}\delta_{v'_n-1,v_n}\\
    \sum_{v^{(2)}_n}\left(2v^{(2)}_n+1\right)\delta_{v'_n,v^{(2)}_n}
       \sqrt{v_n}\delta_{v^{(2)}_n,v_n-1} =&
       \left(2v_n-1\right)\sqrt{v_n}\delta_{v'_n+1,v_n}\\
    \sum_{v^{(2)}_n}\sqrt{\left(v^{(2)}_n-1\right)\left(v^{(2)}_n\right)}
       \delta_{v'_n+2,v^{(2)}_n}\sqrt{v_n+1}\delta_{v^{(2)}_n,v_n+1} =&
       \left(v_n+1\right)\sqrt{v_n}\delta_{v'_n+1,v_n}\\
    \sum_{v^{(2)}_n}\sqrt{\left(v^{(2)}_n-1\right)\left(v^{(2)}_n\right)}
       \delta_{v'_n+2,v^{(2)}_n}\sqrt{v_n}\delta_{v^{(2)}_n,v_n-1} =&
       \sqrt{v_n\left(v_n-1\right)\left(v_n-2\right)}\delta_{v'_n+3,v_n}

The second and third terms both have :math:`\delta_{v'_n-1,v_n}` and sum to
:math:`3\left(v_n+1\right)^{3/2}`; the fourth and fifth terms both have
:math:`\delta_{v'_n+1,v_n}` and sum to :math:`3v_n^{3/2}`. The result is:

.. math::

   \mathcal{Q}^{(3)}_{v'_n,v_n} =&
    \left(\frac{\hbar}{2\omega_n}\right)^{3/2}\left[
    \sqrt{(v_n+3)(v_n+2)(v_n+1)}\delta_{v'_n-3,v_n}+
    3\left(v_n+1\right)^{3/2}\delta_{v'_n-1,v_n}+\right.\\
    &\left.
    3v_n^{3/2}\delta_{v'_n+1,v_n}+
    \sqrt{v_n\left(v_n-1\right)\left(v_n-2\right)}\delta_{v'_n+3,v_n}
    \right]

Fourth Power
============

Inserting the resolution of the identity between :math:`\op{Q}_n^3` and
:math:`\op{Q}_n`, and using the cubed result:

.. math::

   \mathcal{Q}^{(4)}_{v'_n,v_n}\equiv&\braket{v'_n\mid \op{Q}_n^4\mid v_n} =
      \sum_{v^{(2)}_n}\braket{v'_n\mid \op{Q}_n^3\mid v^{(2)}_n}
      \braket{v^{(2)}_n\mid \op{Q}_n\mid v_n}\\
      =& \left(\frac{\hbar}{2\omega_n}\right)^{2}\sum_{v^{(2)}_n}
       \left[\sqrt{(v^{(2)}_n+3)(v^{(2)}_n+2)(v^{(2)}_n+1)}
       \delta_{v'_n-3,v^{(2)}_n}+\right.\\
       &3\left(v^{(2)}_n+1\right)^{3/2}\delta_{v'_n-1,v^{(2)}_n}+
         3\left(v^{(2)}_n\right)^{3/2}\delta_{v'_n+1,v^{(2)}_n}+\\
       &\left.\sqrt{v^{(2)}_n\left(v^{(2)}_n-1\right)\left(v^{(2)}_n-2\right)}
       \delta_{v'_n+3,v^{(2)}_n}\right]\\
       &\left[\sqrt{v_n+1}\delta_{v^{(2)}_n,v_n+1} +
       \sqrt{v_n}\delta_{v^{(2)}_n,v_n-1}\right]

The eight cross terms:

.. math::

    \sum_{v^{(2)}_n}\sqrt{(v^{(2)}_n+3)(v^{(2)}_n+2)(v^{(2)}_n+1)}
       \delta_{v'_n-3,v^{(2)}_n}\sqrt{v_n+1}\delta_{v^{(2)}_n,v_n+1} =&
       \sqrt{(v_n+4)(v_n+3)(v_n+2)(v_n+1)}\delta_{v'_n-4,v_n}\\
    \sum_{v^{(2)}_n}\sqrt{(v^{(2)}_n+3)(v^{(2)}_n+2)(v^{(2)}_n+1)}
       \delta_{v'_n-3,v^{(2)}_n}\sqrt{v_n}\delta_{v^{(2)}_n,v_n-1} =&
       v_n\sqrt{(v_n+2)(v_n+1)}\delta_{v'_n-2,v_n}\\
    \sum_{v^{(2)}_n}3\left(v^{(2)}_n+1\right)^{3/2}\delta_{v'_n-1,v^{(2)}_n}
       \sqrt{v_n+1}\delta_{v^{(2)}_n,v_n+1} =&
       3\left(v_n+2\right)\sqrt{(v_n+2)(v_n+1)}\delta_{v'_n-2,v_n}\\
    \sum_{v^{(2)}_n}3\left(v^{(2)}_n+1\right)^{3/2}\delta_{v'_n-1,v^{(2)}_n}
       \sqrt{v_n}\delta_{v^{(2)}_n,v_n-1} =&
       3v_n^2\delta_{v'_n,v_n}\\
    \sum_{v^{(2)}_n}3\left(v^{(2)}_n\right)^{3/2}\delta_{v'_n+1,v^{(2)}_n}
       \sqrt{v_n+1}\delta_{v^{(2)}_n,v_n+1} =&
       3\left(v_n+1\right)^2\delta_{v'_n,v_n}\\
    \sum_{v^{(2)}_n}3\left(v^{(2)}_n\right)^{3/2}\delta_{v'_n+1,v^{(2)}_n}
       \sqrt{v_n}\delta_{v^{(2)}_n,v_n-1} =&
       3\left(v_n-1\right)\sqrt{v_n\left(v_n-1\right)}\delta_{v'_n+2,v_n}\\
    \sum_{v^{(2)}_n}
       \sqrt{v^{(2)}_n\left(v^{(2)}_n-1\right)\left(v^{(2)}_n-2\right)}
       \delta_{v'_n+3,v^{(2)}_n}\sqrt{v_n+1}\delta_{v^{(2)}_n,v_n+1} =&
       \left(v_n+1\right)\sqrt{v_n\left(v_n-1\right)}\delta_{v'_n+2,v_n}\\
    \sum_{v^{(2)}_n}
       \sqrt{v^{(2)}_n\left(v^{(2)}_n-1\right)\left(v^{(2)}_n-2\right)}
       \delta_{v'_n+3,v^{(2)}_n}\sqrt{v_n}\delta_{v^{(2)}_n,v_n-1} =&
       \sqrt{v_n\left(v_n-1\right)\left(v_n-2\right)\left(v_n-3\right)}
       \delta_{v'_n+4,v_n}

The second and third terms both have :math:`\delta_{v'_n-2,v_n}` and sum to
:math:`\left(4v_n+6\right)\sqrt{(v_n+2)(v_n+1)}`; the fourth and fifth terms
both have :math:`\delta_{v'_n,v_n}` and sum to :math:`6v_n^2+6v_n+3`; the
sixth and seventh terms both have :math:`\delta_{v'_n+2,v_n}` and sum to
:math:`\left(4v_n-2\right)\sqrt{v_n\left(v_n-1\right)}`. The result is:

.. math::

   \mathcal{Q}^{(4)}_{v'_n,v_n} =&
    \left(\frac{\hbar}{2\omega_n}\right)^{2}\left[
    \sqrt{(v_n+4)(v_n+3)(v_n+2)(v_n+1)}\delta_{v'_n-4,v_n}+
    \left(4v_n+6\right)\sqrt{(v_n+2)(v_n+1)}\delta_{v'_n-2,v_n}+\right.\\
    &\left(6v_n^2+6v_n+3\right)\delta_{v'_n,v_n}+
    \left(4v_n-2\right)\sqrt{v_n\left(v_n-1\right)}\delta_{v'_n+2,v_n}+\\
    &\left.
    \sqrt{v_n\left(v_n-1\right)\left(v_n-2\right)\left(v_n-3\right)}
    \delta_{v'_n+4,v_n}\right]
