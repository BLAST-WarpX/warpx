.. _multiphysics-ionization:

Ionization
==========

Field Ionization
----------------

Under the influence of a sufficiently strong external electric field atoms become ionized.
Particularly the dynamics of interactions between ultra-high intensity laser pulses and matter, e.g., Laser-Plasma Acceleration (LPA) with ionization injection, or Laser-Plasma Interactions with solid density targets (LPI) can depend on field ionization dynamics as well.

WarpX models field ionization based on a description of the Ammosov-Delone-Krainov model:cite:p:`mpion-Ammosov1986` following :cite:t:`mpion-ChenPRSTAB13`.

Implementation Details and Assumptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    The current implementation makes the following assumptions

    * Energy for ionization processes is not removed from the electromagnetic fields
    * Only one single-level ionization process can occur per macroparticle and time step
    * Ionization happens at the beginning of the PIC loop before the field solve
    * Angular momentum quantum number :math:`l = 0` and magnetic quantum number :math:`m = 0`

The model implements the following equations (assumptions to :math:`l` and :math:`m` have already been applied).

The electric field amplitude is calculated in the particle's frame of reference.

.. math::

    \begin{aligned}
        \vec{E}_\mathrm{dc} &= \sqrt{ - \frac{1}{\mathrm{c}^2} \left( \vec{u} \cdot \vec{E} \right)^2
                          + \left( \gamma \vec{E} + \vec{u} \times \vec{B} \right)^2 }
        \\
        \gamma &= \sqrt{1 + \frac{\vec{u}^2}{\mathrm{c}^2}}
    \end{aligned}

Here, :math:`\vec{u} = (u_x, u_y, u_z)` is the momentum normalized to the particle mass, :math:`u_i = (\beta \gamma)_i \mathrm{c}`.
:math:`E_\mathrm{dc} = |\vec{E}_\mathrm{dc}|` is the DC-field in the frame of the particle.

.. math::

    \begin{aligned}
        P &= 1 - \mathrm{e}^{-W\mathrm{d}\tau/\gamma}
        \\
        W &= \omega_\mathrm{a} \mathcal{C}^2_{n^* l^*} \frac{U_\mathrm{ion}}{2 U_H}
                \left[ 2 \frac{E_\mathrm{a}}{E_\mathrm{dc}} \left( \frac{U_\mathrm{ion}}{U_\mathrm{H}} \right)^{3/2} \right]^{2n^*-1}
                \times \exp\left[ - \frac{2}{3} \frac{E_\mathrm{a}}{E_\mathrm{dc}} \left( \frac{U_\mathrm{ion}}{U_\mathrm{H}} \right)^{3/2} \right]
        \\
        \mathcal{C}^2_{n^* l^*} &= \frac{2^{2n^*}}{n^* \Gamma(n^* + l^* + 1) \Gamma(n^* - l^*)}
    \end{aligned}

where :math:`\mathrm{d}\tau` is the simulation timestep, which is divided by the particle :math:`\gamma` to account for time dilation. The quantities are: :math:`\omega_\mathrm{a}`, the atomic unit frequency, :math:`U_\mathrm{ion}`, the ionization potential, :math:`U_\mathrm{H}`, Hydrogen ground state ionization potential, :math:`E_\mathrm{a}`, the atomic unit electric field, :math:`n^* = Z \sqrt{U_\mathrm{H}/U_\mathrm{ion}}`, the effective principal quantum number (*Attention!* :math:`Z` is the ionization state *after ionization*.) , :math:`l^* = n_0^* - 1`, the effective orbital quantum number.

Empirical Extension to Over-the-Barrier Regime for Hydrogen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For hydrogen, WarpX offers the modified empirical ADK extension to the Over-the-Barrier (OTB) published in :cite:t:`mpion-zhang_empirical_2014` Eq. (8) (note there is a typo in the paper and there should not be a minus sign in Eq. 8).

.. math::

    W_\mathrm{M} = \exp\left[ a_1 \frac{E^2}{E_\mathrm{b}} + a_2 \frac{E}{E_\mathrm{b}} + a_3 \right] W_\mathrm{ADK}

The parameters :math:`a_1` through :math:`a_3` are independent of :math:`E` and can be found in the same reference. :math:`E_\mathrm{b}` is the classical Barrier Suppresion Ionization (BSI) field strength :math:`E_\mathrm{b} = U_\mathrm{ion}^2 / (4 Z)` given here in atomic units (AU). For a detailed description of conversion between unit systems consider the book by :cite:t:`mpion-Mulser2010`.

Testing
^^^^^^^

* `Testing the field ionization module <../../../../en/latest/usage/examples/field_ionization/README.html>`_.

.. bibliography::
    :keyprefix: mpion-
