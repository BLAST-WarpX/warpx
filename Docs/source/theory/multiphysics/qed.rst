.. _multiphysics-qed:

Quantum Electrodynamics (QED)
=============================

WarpX implements several QED processes:

* Linear Compton scattering
* Linear Breit-Wheeler pair production
* Nonlinear Compton scattering (a.k.a. Quantum synchrotron radiation)
* Nonlinear Breit-Wheeler pair production

The linear processes are single-particle-to-single-particle collisions, while the nonlinear processes involve multiple-particle interactions.
The nonlinear processes are relavent in strong-field QED, e.g. when modeling high-intensity laser-plasma interactions, astrophysical phenomena, and beam-beam physics.
The theoretical foundations and cross-sections for these processes can be found in standard quantum electrodynamics textbooks and specialized references on strong-field QED.
See the bibliography below for a list of useful references.


.. _multiphysics-qed-linear-compton:

Linear Compton Scattering
-------------------------

Linear Compton scattering is usually referred to simply as Compton scattering.
This is the process by which a photon scatters off a free electron/positron, resulting in a change in the photon and electron/positron's energy and direction:

.. math::

   e^- \gamma \rightarrow e^- \gamma \quad \text{or} \quad e^+  \gamma \rightarrow e^+  \gamma

Paragraph 86 in :cite:t:`qed-LandauQED` describes the process in detail.
The differential and total cross-sections are given by the Klein-Nishina formulas (equations 86.10 and 86.16) in the book.

For a photon with initial energy :math:`\hbar \omega_i` scattering off an electron at rest, the scattered photon energy :math:`\hbar \omega_f` is:

.. math::

   \hbar \omega_f = \frac{\hbar \omega_i}{1 + \frac{\hbar \omega_i}{m_e c^2}(1 - \cos \theta)}

where :math:`\theta` is the scattering angle.

The differential cross-section (formula 86.10) is:

.. math::

   \frac{d\sigma}{d\Omega} = \frac{r_e^2}{2} \frac{1}{[1 + x(1 - \cos \theta)]^2} \left[\frac{1}{1 + x(1 - \cos \theta)} + 1 + x(1 - \cos \theta) - \sin^2 \theta \right]

where :math:`r_e = e^2/(4\pi \epsilon_0 m_e c^2)` is the classical electron radius, and :math:`x = \hbar \omega_i/(m_e c^2)` is the ratio of the photon energy to the electron rest mass energy in the rest frame of the electron.

.. math::

   \frac{d\sigma}{d\Omega} = \frac{r_e^2}{2} \left(\frac{\omega_f}{\omega_i}\right)^2 \left(\frac{\omega_f}{\omega_i} + \frac{\omega_i}{\omega_f} - \sin^2 \theta \right)

where :math:`r_e = e^2/(4\pi \epsilon_0 m_e c^2)` is the classical electron radius, and :math:`\omega_i` and :math:`\omega_f` are the initial and final photon frequencies, respectively.


The total cross-section (formula 86.16) is obtained by integrating over all solid angles:

.. math::

   \sigma = \pi r_e^2 \left[ f_1(k) - f_2(k) \right]

where :math:`k = p/(m_e c) = x` is the photon momentum normalized by :math:`m_e c`, and:

.. math::

   f_1(k) = \frac{2(2 + k(1+k)(8+k))}{k^2 (1 + 2k)^2}

   f_2(k) = \frac{(2 + k(2-k)) \ln(1 + 2k)}{k^3}

In the low-energy limit (:math:`k \ll 1`), the total cross-section reduces to the classical Thomson cross-section:

.. math::

   \sigma \approx \frac{8\pi r_e^2}{3} = \sigma_T

In the high-energy limit (:math:`k \gg 1`), the total cross-section decreases approximately as:

.. math::

   \sigma \approx \frac{\pi r_e^2}{k} \left( \ln(2k) + \frac{1}{2} \right)


Using the relation :math:`\omega_f/\omega_i = 1/[1 + x(1 - \cos \theta)]` where :math:`x = \hbar \omega_i/(m_e c^2)`, the differential cross-section can be expressed as:


In WarpX, we model the linear Compton scattering process as a binary collision process of type ``linear_compton`` between a photon species and a electron or positron species.
The implementation is an adapatation of the algorithm described in :cite:t:`qed-HigginsonJCP2019` for nuclear fusion reactions with weighted macroparticles.


This type of collision occurs at the cell level, meaning that two macroparticles can collide only when their centroids are located in the same cell.
WarpX calculates the scattering probability of collision by sampling the total Klein-Nishina cross-section in the rest frame of the lepton.
If the event occurs, the differential cross section is sampled to update the photon's momentum.
Then, the electron momentum is updated accordingly.

.. _multiphysics-qed-linear-breit-wheeler:

Linear Breit-Wheeler pair production
------------------------------------

Linear Breit-Wheeler pair production is the process by which two photons collide to create an electron-positron pair:

.. math::

   \gamma \gamma \rightarrow e^- e^+

This process was first theoretically predicted by Breit and Wheeler in 1934 :cite:t:`qed-Breit1934`.
For pair production to occur, the total energy of the two photons in the center-of-momentum (COM) frame must exceed twice the electron rest mass energy:

.. math::

   \sqrt{(\hbar \omega_1 + \hbar \omega_2)^2 - (\hbar \mathbf{k}_1 + \hbar \mathbf{k}_2)^2 c^2} \geq 2 m_e c^2

where :math:`\hbar \omega_1, \hbar \omega_2` are the photon energies and :math:`\mathbf{k}_1, \mathbf{k}_2` are their wave vectors.

The total cross-section for linear Breit-Wheeler pair production is given by :cite:t:`qed-Gould1967`:

.. math::

   \sigma = \frac{\pi r_e^2}{2} (1 - \beta^2) \left[ (3 - \beta^4) \ln\left(\frac{1 + \beta}{1 - \beta}\right) + 2\beta(\beta^2 - 2) \right]

where :math:`r_e = e^2/(4\pi \epsilon_0 m_e c^2)` is the classical electron radius, and :math:`\beta` is the velocity of the electron (or positron) in the COM frame:

.. math::

   \beta = \sqrt{1 - \frac{1}{s}}, \quad s = \left(\frac{E^*}{m_e c^2}\right)^2

Here, :math:`E^*` is the kinetic energy of each photon in the COM frame. If :math:`E^* < m_e c^2`, the total kinetic energy in the COM frame is insufficient to produce a pair, and the cross-section is zero.

The differential cross-section describes the angular distribution of the produced electron-positron pairs.
The differential cross-section for linear Breit-Wheeler pair production is given by :cite:t:`qed-Gould1967`:

.. math::

   \frac{d\sigma}{d\Omega} = \frac{r_e^2}{2} \frac{1 - \beta^4 (1 - \cos^2\theta)^2}{(1 - \beta^2 \cos^2\theta)^2}

where :math:`\theta` is the angle between the electron momentum and the direction of one of the incident photons in the COM frame, and :math:`\beta` is the velocity of the electron (or positron) in the COM frame as defined above.

In the current implementation, the angular distribution is isotropic in the COM frame, meaning the pairs are emitted uniformly in all directions.
This is an approximation; the correct angular distribution is more complex and depends on the photon energies and collision geometry :cite:t:`qed-Ribeyre2018`.
A work-in-progress implementation of the differential cross-section sampling is available in PR #6141 .
For the complete theoretical treatment, see :cite:t:`qed-JauchRohrlich` and :cite:t:`qed-Gould1967`.

Similar to linear Compton scattering, linear Breit-Wheeler is implemented as a binary collision process of type ``linear_breit_wheeler`` between two photon species.
The implementation follows the same numerical algorithm as that of fusion reactions :cite:t:`qed-HigginsonJCP2019` for weighted macroparticles.

This type of collision occurs at the cell level, meaning that two macroparticles can collide only when their centroids are located in the same cell.
WarpX calculates the pair-production probability by sampling the total Breit-Wheeler cross-section in the COM frame.
If the event occurs, four outgoing macroparticles are created instead of two to conserve charge locally:
two positrons and two electrons.
One positron and one electron are generated at the position of the first photon, and the other positron and electron are generated at the position of the second photon.
This ensures local charge conservation at each photon's location.
The momentum of the created particles is determined by energy and momentum conservation, with the angular distribution currently assumed to be isotropic in the COM frame.



Nonlinear Compton Scattering
----------------------------

Nonlinear Compton scattering (also known as quantum synchrotron radiation or quantum synchrotron emission) is the process by which relativistic electrons or positrons emit high-energy photons when traversing strong electromagnetic fields.
This process is a quantum extension of classical synchrotron radiation, where the particle's trajectory is curved by the electromagnetic field, leading to photon emission.

The quantum nature becomes important when the emitted photon energy is comparable to the electron's energy, requiring a quantum mechanical treatment.
The emission probability depends on the quantum parameter :math:`\chi_e`, defined as:

.. math::

   \chi_e = \frac{\gamma_e}{E_S} \sqrt{(\mathbf{E} + \mathbf{v} \times \mathbf{B})^2 - (\mathbf{E} \cdot \mathbf{v}/c)^2}

where :math:`\gamma_e` is the electron's Lorentz factor, :math:`E_S = m_e^2 c^3 / (e \hbar)` is the Schwinger critical field (:math:`\approx 1.32 \times 10^{18}` V/m), :math:`\mathbf{E}` and :math:`\mathbf{B}` are the electric and magnetic fields, and :math:`\mathbf{v}` is the electron velocity.

The photon emission rate :math:`W` is given by equation 15 in :cite:t:`qed-Fedeli_2022`:

.. math::
   :label: eq15

   W = \frac{\alpha m_e c^2}{\hbar \gamma} \int_0^{\chi_e} \frac{d\chi_\gamma}{\chi_\gamma} \left[ \frac{2}{\sqrt{3}} K_{2/3}\left( \frac{2\chi_\gamma}{3\chi_e} \right) + \frac{\chi_\gamma}{\chi_e} \int_{2\chi_\gamma/(3\chi_e)}^\infty K_{5/3}(x) dx \right]

where :math:`\alpha = e^2/(4\pi\epsilon_0 \hbar c)` is the fine structure constant, :math:`\gamma` is the Lorentz factor of the electron, :math:`\chi_e` is the quantum parameter for the electron, :math:`\chi_\gamma` is the quantum parameter for the emitted photon, and :math:`K_{2/3}` and :math:`K_{5/3}` are modified Bessel functions of the second kind.

The photon emission rate per unit photon energy :math:`\varepsilon_\gamma` is given by equation 16 in :cite:t:`qed-Fedeli_2022`:

.. math::
   :label: eq16

   \frac{dW}{d\varepsilon_\gamma} = \frac{\alpha m_e c^2}{\hbar \gamma \varepsilon_\gamma} \left[ \frac{2}{\sqrt{3}} K_{2/3}\left( \frac{2\chi_\gamma}{3\chi_e} \right) + \frac{\chi_\gamma}{\chi_e} \int_{2\chi_\gamma/(3\chi_e)}^\infty K_{5/3}(x) dx \right]

where :math:`\varepsilon_\gamma = \hbar\omega` is the photon energy.

For computational efficiency, lookup tables are pre-computed as a function of :math:`\chi_e` and :math:`\chi_\gamma` to avoid expensive function evaluations during the simulation.

**Implementation in WarpX:**

In WarpX, quantum synchrotron radiation is implemented using lookup tables from the PICSAR library :cite:t:`qed-Fedeli_2022`.
The process is activated by setting ``<species>.do_qed_quantum_sync = 1`` for electron or positron species.
When an electron or positron emits a photon, the photon is added to a specified photon product species, and the emitting particle's energy and momentum are updated accordingly.

The implementation uses a Monte Carlo approach to determine whether photon emission occurs at each time step.
The emission probability is calculated based on the local electromagnetic field strength and the particle's energy, using the quantum parameter :math:`\chi_e`.
If emission occurs, the photon energy is sampled from the appropriate probability distribution using lookup tables.
The lookup tables are pre-computed and stored, allowing for efficient computation during the simulation.
Two lookup tables are used: one for the evolution of the optical depth (used to determine if emission occurs), and another for sampling the photon energy fraction :math:`\xi` given the quantum parameter :math:`\chi_e`.

For more details on the theoretical formulation and implementation, see :cite:t:`qed-Fedeli_2022`.


Nonlinear Breit-Wheeler pair production
---------------------------------------

Nonlinear Breit-Wheeler pair production is the process by which a high-energy photon creates an electron-positron pair when interacting with an intense electromagnetic field.
This is a nonlinear extension of the linear Breit-Wheeler process, where the strong external field provides the necessary energy and momentum for pair creation.

The process becomes significant when the photon's quantum parameter :math:`\chi_\gamma` is large:

.. math::

   \chi_\gamma = \frac{\hbar \omega_\gamma}{m_e c^2} \frac{E}{E_S}

where :math:`\hbar \omega_\gamma` is the photon energy, :math:`E` is the electric field strength, and :math:`E_S = m_e^2 c^3 / (e \hbar)` is the Schwinger critical field (:math:`\approx 1.32 \times 10^{18}` V/m).

The pair production rate :math:`W` is given by equation 17 in :cite:t:`qed-Fedeli_2022`:

.. math::
   :label: eq17

   W = \frac{\alpha m_e c^2}{\hbar \omega_\gamma} \int_0^{\chi_\gamma} \frac{d\chi_e}{\chi_e} \left[ \frac{2}{\sqrt{3}} K_{2/3}\left( \frac{2\chi_e}{3\chi_\gamma} \right) + \frac{\chi_e}{\chi_\gamma} \int_{2\chi_e/(3\chi_\gamma)}^\infty K_{5/3}(x) dx \right]

where :math:`\alpha = e^2/(4\pi\epsilon_0 \hbar c)` is the fine structure constant, :math:`\omega_\gamma` is the photon frequency, :math:`\chi_\gamma` is the quantum parameter for the photon, :math:`\chi_e` is the quantum parameter for the created electron (or positron), and :math:`K_{2/3}` and :math:`K_{5/3}` are modified Bessel functions of the second kind.

The pair production rate per unit electron energy :math:`\varepsilon_e` is given by equation 18 in :cite:t:`qed-Fedeli_2022`:

.. math::
   :label: eq18

   \frac{dW}{d\varepsilon_e} = \frac{\alpha m_e c^2}{\hbar \omega_\gamma \varepsilon_e} \left[ \frac{2}{\sqrt{3}} K_{2/3}\left( \frac{2\chi_e}{3\chi_\gamma} \right) + \frac{\chi_e}{\chi_\gamma} \int_{2\chi_e/(3\chi_\gamma)}^\infty K_{5/3}(x) dx \right]

where :math:`\varepsilon_e` is the energy of the created electron (or positron).

**Optical Depth:**

The optical depth :math:`\tau` is a key quantity used in the Monte Carlo implementation to determine whether pair production occurs.
The optical depth evolves according to:

.. math::

   \frac{d\tau}{dt} = W

where :math:`W` is the pair production rate.
At each time step, the optical depth is evolved, and if it exceeds a randomly sampled threshold, pair production occurs.
The optical depth is reset after each pair production event.
This approach allows for efficient Monte Carlo sampling of the pair production process while maintaining the correct statistical distribution.

For computational efficiency, lookup tables are pre-computed as a function of :math:`\chi_\gamma` and :math:`\chi_e` to avoid expensive function evaluations during the simulation.

**Implementation in WarpX:**

Nonlinear Breit-Wheeler pair production is implemented in WarpX using lookup tables from the PICSAR library :cite:t:`qed-Fedeli_2022`.
The process is activated by setting ``<species>.do_qed_breit_wheeler = 1`` for photon species.
When a photon creates a pair, the electron and positron are added to their respective product species, and the photon is removed from the simulation.

The implementation uses a Monte Carlo approach based on optical depth evolution to determine whether pair creation occurs at each time step.
The pair production probability is calculated based on the photon energy and the local electromagnetic field strength, using the quantum parameter :math:`\chi_\gamma`.
If pair creation occurs, the electron and positron energies are sampled from the appropriate probability distribution using lookup tables.
The lookup tables are pre-computed and stored, allowing for efficient computation during the simulation.
Two lookup tables are used: one for the evolution of the optical depth (used to determine if pair production occurs), and another for sampling the electron energy fraction given the quantum parameter :math:`\chi_\gamma`.

For theoretical details and numerical implementation, refer to :cite:t:`qed-Fedeli_2022`.



Schwinger process
-----------------

If the code is compiled with QED and the user activates the Schwinger process in the input file,
electron-positron pairs can be created in vacuum in the function
``MultiParticleContainer::doQEDSchwinger``:

.. doxygenfunction:: MultiParticleContainer::doQEDSchwinger

``MultiParticleContainer::doQEDSchwinger`` in turn calls the function ``filterCreateTransformFromFAB``:

Filter Create Transform Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``filterCreateTransformFromFAB`` proceeds in three steps.
In the filter phase, we loop on every cell and calculate the number of physical pairs created within
the time step dt as a function of the electromagnetic field at the given cell position.
This probabilistic calculation is done via a wrapper that calls the ``PICSAR`` library.
In the create phase, the particles are created at the desired positions, currently at the cell nodes.
In the transform phase, we assign a weight to the particles depending on the number of physical
pairs created.
At most one macroparticle is created per cell per timestep per species, with a weight corresponding to
the total number of physical pairs created.

So far the Schwinger module requires using :pp:param:`warpx.grid_type = collocated` or
:pp:param:`algo.field_gathering = momentum-conserving` (so that the auxiliary fields are calculated on the nodes)
and is not compatible with either mesh refinement, RZ, RCYLINDER, and RSPHERE coordinates or single precision.






.. bibliography::
