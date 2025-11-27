.. _multiphysics-qed:

Quantum Electrodynamics (QED)
=============================

Quantum synchrotron
-------------------

.. note::

   Section empty!

Breit-Wheeler
-------------

.. note::

   Section empty!

Schwinger process
-----------------

In the Schwinger process, electron-positron pairs are created in vacuum by a sufficiently strong electromagnetic field.
An expression for the Schwinger pair production rate can be found in :cite:p:`mqed-NarozhnyPRA2004`.
Rewritten in SI units, the pair production rate per unit volume is:

.. math::

    \dfrac{d^2N}{dt dV} = \dfrac{e^2 E_s^2}{4 \pi^2 \hbar^2 c} \epsilon \eta \coth{\left({\dfrac{\pi \eta}{\epsilon}}\right)}\exp{\left({-\dfrac{\pi}{\epsilon}}\right)}


where $e$ is the elementary charge, :math:`E_s`` is the Schwinger field, :math:`\epsilon = \mathcal{E}/E_s` and :math:`\eta = \mathcal{H}/E_s`.
:math:`\mathcal{E}` and :math:`\mathcal{H}$` are given by:

.. math::

    \mathcal{E} = \sqrt{\sqrt{\mathcal{F}^2 + \mathcal{G}^2} + \mathcal{F}} \\
    \mathcal{H} = \sqrt{\sqrt{\mathcal{F}^2 + \mathcal{G}^2} - \mathcal{F}} \\

:math:`\mathcal{F}` and :math:`\mathcal{G}$` are the invariants of the electromagnetic field and are equal to:

.. math::

    \mathcal{F} = (\mathbf{E}^2 - c^2 \mathbf{B}^2)/2 \\
    \mathcal{G} = c \mathbf{E} \cdot \mathbf{B} \\

If the user activates the Schwinger process in the input file,
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

So far the Schwinger module requires using ``warpx.grid_type = collocated`` or
``algo.field_gathering = momentum-conserving`` (so that the auxiliary fields are calculated on the nodes)
and is not compatible with either mesh refinement, RZ, RCYLINDER, and RSPHERE coordinates or single precision.

.. bibliography::
    :keyprefix: mqed-
