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

#####################
Problem Specification
#####################

The goal of rovib is to model the rovibrational energy levels of a non-linear
molecule containing |Natom| atoms. The location of the :math:`i`\ th atom in the
molecule is given by the vector :math:`\vec{r}_i` and the location of the
molecule as a whole is given by the matrix of atomic coordinates,
:math:`\tens{r} = \left({\vec{r}}_1, {\vec{r}}_2, \ldots, {\vec{r}}_{\Natom}\right)`.
Similarly,
|Mvec| is the diagonal matrix of atomic masses such that :math:`M_{ii}` is the
mass of the :math:`i`\ th atom. We assume that the Born-Oppenheimer
approximation is valid. In this case, we can assume that the nuclei move on a
single potential energy surface (PES), :math:`V\left(\tens{r}\right)`, defined
by the electronic Hamiltonian. We will also assume that we have located a
minimum energy geometry on the PES, :math:`\rzvec`. As the molecule vibrates
and rotations it will be assumed to have total angular momentum
:math:`\tens{J}`.

Our original goal can be restated as solving the time-independent Schrödinger
equation for the motion of the nuclei on the PES.

*****************
Coordinate System
*****************

Main article: :ref:`coordinate_system`.

If :math:`\tens{r}` and :math:`\tens{r}^0` are defined in the Eckart frame,
then the internal coordinates of the molecule are given by:

.. math::

   \tens{Q} = \Lvec^T\Mvec^{1/2}\left(\tens{r} - \tens{r}^0\right)

where :math:`\Lvec` is the transformation matrix that diagonalizes the mass-
weighted Hessian of the PES at :math:`\tens{r}^0`. The corresponding vibrational
momenta are given by:
:math:`\tens{P} = -i\hbar\frac{\partial}{\partial \tens{Q}}`.

In practice, we will solve the rovibrational Schrödinger equation using second
quantization. In this basis, the vibrational coordinates and momenta take on
the form of creation and annihilation operators, |qvec| and |pvec|, and the
coordinate forms of :math:`\tens{Q}` and :math:`\tens{P}` are needed only to
compute the necessary coefficients for the second-quantized Hamiltonian.

**************************************
Defining the Rovibrational Hamiltonian
**************************************

The total rovibrational Hamiltonian |Hvib| of the molecule is given by:

.. math::

    \Hvib =
       \frac{1}{2}\left(\jvec-\pivec\right)^T\muvec\left(\jvec-\pivec\right) +
       \frac{1}{2}\sum_{i=1}^{\Nvib}\pop_i^2 +
       \Vop(\qvec) -
       \frac{\hbar^2}{8}\tr(\muvec)

Here |pivec| is the angular momentum of the atoms due to their vibrational
motion, it is given by:

.. math::

   \piop_\alpha = \sum_{i,j}^{\Nvib} \zeta_{i,j}^\alpha \qop_i \pop_j

where |zetavec| is the Coriolis coupling constants. |zetavec| is given by:

.. math::

    \zeta_{i,j}^\alpha = \sum_{k=1}^{\Natom}
        \left(\Lvec_{ki}\times\Lvec_{kj}\right)_\alpha.

|muvec| is the inverse of the effective inertia tensor, |Ieffvec|. |Ieffvec| is
given by:

.. math::
    \Ieffop_{\alpha,\beta} = \Iop_{\alpha,\beta} -
       \sum_{i,j,k}^{\Nvib} \zeta^{\alpha}_{i,k}\zeta^\beta_{j,k}
       \qop_i \qop_j

where |Ivec| is the instantaneous inertia tensor of the molecule.

In practice, we rarely have an analytical form of the PES, so we
instead Taylor expand :math:`\Vop(\qvec)` in displacements from the minimum
energy geometry. We set the energy of the minimum energy geometry to zero (so
the zeroth term of the Taylor series is zero). Similarly, since the expansion
point is the minimum energy geometry the gradient is zero (which means 1st order
term is also zero). Truncating the series at the quartic term gives:

.. math::

    \Vop(\qvec) \approx&  \Vharm + \Vanharm\\
    \Vharm \equiv& \frac{1}{2}\sum_{i}^{\Nvib}\omega_i^2 \qop_i^2 \\
    \Vanharm \equiv& \frac{1}{6}\sum_{i,j,k}^{\Nvib}g_{i,j,k}
                       \qop_i \qop_j \qop_k +
               \frac{1}{24}\sum_{i,j,k,\ell}^{\Nvib}h_{i,j,k,\ell}
                 \qop_i \qop_j \qop_k \qop_\ell

where we used the fact that the harmonic force constants are diagonal in the
normal coordinate basis and introduced the third- and fourth-order force
constants as :math:`g_{i,j,k}` and :math:`F_{i,j,k,\ell}` respectively.

Similarly, since |muvec| is also a function of the normal coordinates, it is
also expanded as a fourth-order Taylor series about the minimum energy geometry,
:math:`\muvec \approx \sum_{n=0}^{4}\muvec^{(n)}`.

Inserting the Taylor expansions of :math:`\Vop(\qvec)` and |muvec| into the
vibrational Hamiltonian, and retaining terms up to fourth order in |qvec|
gives:

.. math::

       \Hvib = \HHO + \HRR + \Vanharm  + \HvibCor + \Hrovib +
         \mathcal{O}(\qvec^5)

Where:

- :math:`\HHO \equiv  \frac{1}{2}\sum_{i=1}^{\Nvib}\pop_i^2 + \Vharm` is the
  harmonic oscillator Hamiltonian.
- :math:`\HRR \equiv \frac{1}{2}\jvec^T\muvec^{(0)}\jvec` is the rigid rotor
  Hamiltonian.
- :math:`\Vanharm` is the anharmonic potential energy.

:math:`\HvibCor` is:

.. math::

   \HvibCor\equiv& \frac{1}{2}\sum_{n=0}^{2}\pivec^T\muvec^{(n)}\pivec +
                  \Hwatson\\
   \Hwatson \equiv& -\frac{\hbar^2}{8}\sum_{n=0}^{4}\tr(\muvec^{(n)})

is the vibrational Coriolis Hamiltonian and :math:`\Hrovib` is the rovibrational
coupling Hamiltonian, which is given by:

.. math::

   \Hrovib \equiv \frac{1}{2}\sum_{n=1}^{4}\jvec^T\muvec^{(n)}\jvec -
         \frac{1}{2}\sum_{n=0}^{3}\jvec^T\muvec^{(n)}\pivec-
         \frac{1}{2}\sum_{n=0}^{3}\pivec^T\muvec^{(n)}\jvec

**********************************
Matrix Elements of the Hamiltonian
**********************************

The rovibtrational problem is solved by diagonalizing |Hvib| in a basis set
:math:`\ket{\vvec,J,K}`. The basis functions are the direct product of
:math:`\Nvib` harmonic oscillator eigenfunctions and an angular momentum
eigenfunction.

Harmonic Oscillator Matrix Elements
===================================

The Harmonic Oscillator Hamiltonian, |HHO|, is diagonal in the
:math:`\ket{\vvec,J,K}` basis with matrix elements:

.. math::

    \braket{\vvec',J',K'\mid\HHO\mid\vvec,J,K} =
       \delta_{\vvec',\vvec}\delta_{J',J}\delta_{K',K}
       \sum_{i=1}^{\Nvib}\hbar\omega_i\left(v_i + \frac{1}{2}\right)

Rigid Rotor Matrix Elements
===========================

Main article: :ref:`rigid_rotor`.

In the :math:`\ket{\vvec,J,K}` basis, the rigid rotor Hamiltonian, |HRR|, is
diagonal in :math:`\vvec` and :math:`J`, but not :math:`K`. The matrix elements
are:

.. math::

   \braket{\vvec',J',K'\mid \HRR\mid\vvec,J,K} =
      \delta_{\vvec',\vvec}\delta_{J',J}\left(
      T^{(\text{RR},0)}_{J,K',K} +
      T^{(\text{RR},-2)}_{J,K',K} +
      T^{(\text{RR},2)}_{J,K',K}\right)

where:

.. math::

   T^{(\text{RR},0)}_{J,K',K} \equiv& \delta_{K',K}\left[
                                      \frac{\gamma_x+\gamma_y}{2}
                                      \left(J(J+1) - K^2\right) +
                                      \gamma_z K^2\right]\\
   T^{(\text{RR},-2)}_{J,K',K} \equiv& \delta_{K',K-2}\left[
                                      \frac{\gamma_x-\gamma_y}{4}
                                      C^-_{J,K}C^-_{J,K-1}\right]\\
   T^{(\text{RR},2)}_{J,K',K} \equiv& \delta_{K',K+2}\left[
                                      \frac{\gamma_x-\gamma_y}{4}
                                      C^+_{J,K}C^+_{J,K+1}\right]

with :math:`\gamma_x`, :math:`\gamma_y`, and :math:`\gamma_z` being the
rotational constants of the molecule and:

.. math::

   C^\pm_{J,K} = \sqrt{J(J+1) - K(K\pm1)}.

Anharmonic Matrix Elements
==========================

Main article: :ref:`anharmonic_potential`.

In the :math:`\ket{\vvec,J,K}` basis, the anharmonic potential energy,
:math:`\Vanharm` is:

.. math::

   \braket{\vvec', J', K' \mid \Vanharm\mid\vvec, J, K} =&
    \delta_{J',J}\delta_{K',K}\left[
       \sum_{i<j<k}^{\Nvib} g_{ijk}T^{(111)}_{ijk}\right.\\
    &+ \frac{1}{2}\sum_{i\neq j}^{\Nvib} g_{iij}T^{(210)}_{ij}\\
    &+ \frac{1}{3!}\sum_{i=1}^{\Nvib} g_{iii}T^{(300)}_{i}\\
    &+ \sum_{i<j<k<l}^{\Nvib} h_{ijkl}T^{(1111)}_{ijkl}\\
    &+ \frac{1}{2}\sum_{i=1}^{\Nvib}\sum_{\substack{j<k\\ j,k\neq i}}^{\Nvib}
       h_{iijk}T^{(211)}_{ijk}\\
    &+ \frac{1}{4}\sum_{i<j}^{\Nvib} h_{iijj}T^{(22)}_{ij}\\
    &+ \frac{1}{3!}\sum_{i\neq j}^{\Nvib} h_{iiij}T^{(31)}_{ij}\\
    &+ \left.\frac{1}{4!}\sum_{i=1}^{\Nvib} h_{iiii}T^{(4)}_{i}\right]

Vibrational Coriolis Matrix Elements
====================================

Main article: :ref:`vibration_coriolis`.

.. math::

   \braket{\vvec',J',K'\mid \HvibCor\mid\vvec,J,K} =&
      \delta_{J',J}\delta_{K',K}\braket{\vvec'\mid \HvibCor\mid\vvec}\\
   \braket{\vvec'\mid \HvibCor\mid\vvec}=&
   \frac{1}{2}\sum_{n=0}^{2}\sum_{\alpha,\beta=1}^{3}
      \sum_{i,j,k,\ell=1}^{\Nvib}\sum_{\midx_n}
      \zeta^\alpha_{i,j}\zeta^\beta_{k,\ell}
      \eta^{\midx_n}_{\alpha\beta}
      \braket{\vvec'\mid \qop_i \pop_j \qop_{\midx_n} \qop_k
              \pop_\ell \mid\vvec} -\\
      & \frac{\hbar^2}{8}\sum_{n=0}^{4}
       \sum_{\alpha=1}^{3}\sum_{\midx_n}
      \eta^{\midx_n}_{\alpha\alpha}\braket{\vvec'\mid \qop_{\midx_n}\mid\vvec}

Rovibrational Coupling Matrix Elements
======================================

Main article: :ref:`rovibrational_coupling`.
