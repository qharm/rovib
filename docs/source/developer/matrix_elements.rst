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


.. _matrix_elements:

##########################
Evaluating Matrix Elements
##########################

The bulk of the work of this package is evaluating the matrix elements of the
Hamiltonian. This page focuses on the details of how the matrix elements will
be evaluated.

**************
Considerations
**************

#. Hamiltonian is a sum of terms.

   - The design calls for one class per Hamiltonian term.
   - The matrix elements for the individual terms will be evaluated term by
     term, and then summed to get the total Hamiltonian matrix element.

#. Basis functions are assumed to be products of harmonic oscillator
   wavefunctions and angular momentum eigenstates.

#. The Hamiltonian will be sparse in the basis set.

   - Whether it's sparse enough to use sparse matrix libraries is TBD.
   - Initial implementation will focus on filling in the non-zero elements of
     a dense matrix.
   - The functions that build matrices only visit the elements that may be
     non-zero.

#. The harmonic oscillator, anharmonic potential, vibrational coriolis, and
   rovibrational coupling terms are all products of vibrational operators. We
   need a mechanism to evaluate matrix elements of products of vibrational
   operators.

   - The final matrix elements are direct products of per-mode matrix elements;
     however, the per-mode matrices are extremely sparse.


***********************
Full Basis: Algorithm 1
***********************

This algorithm assumes we are building the full Hamiltonian matrix in a dense
format.

#. Input is the Hamiltonian ``H`` and the basis set ``B``.

#. Loop over Hamiltonian terms, ``t``.

#. Build the matrix of ``t``. Its matrix is a sum of direct products of a
   matrix in the vibrational basis and a matrix in the rotational basis,
   which are evaluated with the algorithms below.

#. Add the matrix of ``t`` to the Hamiltonian matrix.

******************************
Vibrational Basis: Algorithm 1
******************************

We need to be able to on-the-fly build elements like
:math:`c\braket{\vvec'\mid \op{O}^n \mid\vvec}`, where :math:`\op{O}^n` is a
product of :math:`n` vibrational operators and :math:`c` is a coefficient whose
value depends on the modes :math:`\op{O}^n` operates on. The input to our
function will be ``bras``, the ``VibrationalAnsatz`` of allowed bra states,
``ops``, the list of operators, ``kets``, the ``VibrationalAnsatz`` of allowed
ket states, and ``reducer`` a function that combines the coefficients. This is
implemented by
:func:`rovib.matrix_elements.vibrational_matrix_elements.vibrational_matrix`.
The elements in the full basis follow from the direct product of the result
with the matrix of the rotational part of the term.

Elements in ``ops`` are assumed to be ``HarmonicOperator``,
``PotentialOperator``, ``CoriolisOperator``, or ``WatsonOperator`` objects.

The first three steps do not depend on the ket, so they are done once.

#. Let ``op_range[i] = ops[i].vibrational_operators()``. Loop over the
   Cartesian product of ``op_range``. Let ``op_list`` be the current list of
   operators.

#. Let ``c[i] = ops[i].coefficient(op_list[i])``. Call ``reducer(c)`` to get
   the coefficient for this term. If it is zero, skip to the next ``op_list``.

#. Operators on different modes commute, so group ``op_list`` by mode, keeping
   the order of the operators within each mode. Let ``word[i]`` be the
   operators acting on mode ``i``. Modes with no operators are left out.

The remaining steps are done for each ket.

#. Loop over the states in ``kets``. Let ``ket`` be the current ket state
   and ``values`` be an empty map from bra states to matrix elements.

#. For each term from steps 1-3, and each mode ``i`` in ``word``, apply
   ``word[i]`` to the ket's state of mode ``i``. This gives the allowed values
   of :math:`v'_i` for :math:`\braket{v'_i\mid \op{O}^{(i)}\mid v_i}`, where
   :math:`\op{O}^{(i)}` is the product in ``word[i]``. For example, if
   ``word[i]`` has 3 operators then the allowed values of :math:`v'_i` are such
   that :math:`v'_i - v_i` is one of -3, -1, 1, or 3, and states with
   :math:`v'_i < 0` never appear.

#. Shifts in each mode are independent, so loop over the Cartesian product of
   the results of the previous step. Let ``ket_dp`` be ``ket`` with each mode
   in ``word`` replaced by the current :math:`v'_i`. If ``ket_dp`` is not in
   ``bras``, skip to the next shift.

#. Add ``c`` times the product of the per-mode matrix elements to
   ``values[ket_dp]``. Different terms can couple the same bra and ket, so the
   contributions must be summed rather than assigned.

#. Once all terms are done, set ``H[bra, ket] = values[bra]`` for each ``bra``
   in ``values``.

*****************************
Rotational Basis: Algorithm 1
*****************************

The rotational analog of the above builds :math:`\braket{J',K'\mid\op{O}^n\mid J,K}`,
where :math:`\op{O}^n` is a product of :math:`n` ``JOperator`` objects. The input
to our function will be ``bras``, the ``RotationalAnsatz`` of allowed bra states,
``ops``, the list of ``JOperator`` objects, and ``kets``, the ``RotationalAnsatz``
of allowed ket states. This is implemented by
:func:`rovib.matrix_elements.rotational_matrix_elements.rotational_matrix`.

Each element of ``ops`` is a single operator, not a sum, so no ``reducer`` is
needed; coefficients are applied by the caller. The algorithm is:

#. If ``bras`` and ``kets`` have different values of :math:`J`, the matrix is
   zero, because ``JOperator`` objects conserve :math:`J`.

#. Loop over the states in ``kets``. Let ``ket`` be the current ket state.

#. Apply ``ops`` to the :math:`K` of ``ket``, the rightmost operator first, and
   sum the amplitudes of each resulting value of :math:`K'`. Set the element for
   each :math:`K'` to its summed amplitude.

The rigid rotor Hamiltonian is then

.. math::

   \braket{\vvec',J',K'\mid\HRR\mid\vvec,J,K} =
      \braket{\vvec'\mid\vvec}\sum_{\alpha}\gamma_\alpha
      \braket{J',K'\mid\jop_\alpha^2\mid J,K}

where :math:`\braket{\vvec'\mid\vvec}` is ``vibrational_matrix`` with no
operators.
