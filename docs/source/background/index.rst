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

######################
Theoretical Background
######################

Provides notes on the theoretical background of the rovibrational problem
including the relevant equations. Notes are based on DOI: 10.1063/5.0333947.

********
Notation
********

The following typesetting conventions are used:

- Scalars are in normal font, e.g., |Natom|.
- Tensors (of rank one or higher) are bold, e.g., |zetavec|. When a tensor is
  written in element form the elements are scalars, and are therefore in
  normal font, e.g., :math:`\zeta^\alpha_{i,j}`.
- Operators have hats, e.g., |Hvib|. Tensor operators are also bold, e.g.,
  |jvec|, but their elements are not, e.g., :math:`\jop_\alpha`.
- Anything that depends on the nuclear coordinates or momenta is an operator.

.. toctree::
   :maxdepth: 2

   problem_definition
   coordinate_system
   watson_mu
   angular_wavefunction
   vibrational_wavefunction
   rigid_rotor
   anharmonic_potential
   vibration_coriolis
   rovibrational_coupling
