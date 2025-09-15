:orphan:

WarpX
-----

WarpX is an advanced **Particle-In-Cell** code.

It supports many features including:

    - Multiple types of field solvers (incl. for `Maxwell's equations <https://warpx.readthedocs.io/en/latest/theory/pic.html#field-solve>`__, Poisson's equation, and `Maxwell-Ampere coupled with Ohm's law <https://warpx.readthedocs.io/en/latest/theory/kinetic_fluid_hybrid_model.html>`__)
    - Various grid geometries (1D/2D/3D Cartesian, cylindrical, spherical)
    - Multi-physics packages (incl. ionization, atomic, fusion and collisional physics, as well as quantum electrodynamics)
    - Advanced numerical methods (incl. explicit and implicit time advance, mesh refinement, boosted-frame simulations, embedded boundaries, pseudo-spectral solvers)

For details on these features, see the :ref:`theory section <theory>`.
WarpX has been applied to a wide variety of science projects, see :ref:`highlights <highlights>`.

In addition, WarpX is a *highly-parallel and highly-optimized code*:

    - Can run on multi-core CPUs as well as NVIDIA, AMD or Intel GPUs
    - Scales to the world's largest supercomputers and includes load balancing capabilities. WarpX was awarded the `2022 ACM Gordon Bell Prize <https://www.exascaleproject.org/ecp-supported-collaborative-teams-win-the-2022-acm-gordon-bell-prize-and-special-prize/>`__.
    - Multi-platform code that can run on Linux, macOS and Windows.
    - Can be run and extended via its Python interface, e.g., to couple to other codes or AI/ML frameworks.

.. _contact:

Contact us
^^^^^^^^^^

If you are starting using WarpX, or if you have a user question, please pop in our `discussions page <https://github.com/BLAST-WarpX/warpx/discussions>`__ and get in touch with the community.

The `WarpX GitHub repo <https://github.com/BLAST-WarpX/warpx>`__ is the main communication platform.
Have a look at the action icons on the top right of the web page: feel free to watch the repo if you want to receive updates, or to star the repo to support the project.
For bug reports or to request new features, you can also open a new `issue <https://github.com/BLAST-WarpX/warpx/issues>`__.

We also have a `discussion page <https://github.com/BLAST-WarpX/warpx/discussions>`__ on which you can find already answered questions, add new questions, get help with installation procedures, discuss ideas or share comments.

.. raw:: html

   <style>
   /* front page: hide chapter titles
    * needed for consistent HTML-PDF-EPUB chapters
    */
   section#installation,
   section#usage,
   section#tutorials,
   section#theory,
   section#data-analysis,
   section#development,
   section#maintenance,
   section#epilogue {
       display:none;
   }
   </style>

.. toctree::
   :hidden:

   coc
   acknowledge_us
   highlights

Installation
------------
.. toctree::
   :caption: INSTALLATION
   :maxdepth: 1
   :hidden:

   install/users
   install/cmake
   install/hpc
..   install/changelog
..   install/upgrade

Usage
-----
.. toctree::
   :caption: USAGE
   :maxdepth: 1
   :hidden:

   usage/how_to_run
   usage/examples
   usage/python
   usage/parameters
   usage/workflows
   usage/faq

Tutorials
---------
.. toctree::
   :caption: TUTORIALS
   :maxdepth: 1
   :hidden:

   tutorials

Data Analysis
-------------
.. toctree::
   :caption: DATA ANALYSIS
   :maxdepth: 1
   :hidden:

   dataanalysis/formats
   dataanalysis/yt
   dataanalysis/openpmdviewer
   dataanalysis/openpmdapi
   dataanalysis/paraview
   dataanalysis/visit
   dataanalysis/visualpic
   dataanalysis/picviewer
   dataanalysis/reduced_diags
   dataanalysis/workflows

Theory
------
.. toctree::
   :caption: THEORY
   :maxdepth: 1
   :hidden:

   theory/intro
   theory/pic
   theory/amr
   theory/boundary_conditions
   theory/boosted_frame
   theory/multiphysics_extensions
   theory/kinetic_fluid_hybrid_model
   theory/cold_fluid_model

Development
-----------
.. toctree::
   :caption: DEVELOPMENT
   :maxdepth: 1
   :hidden:

   developers/contributing
   developers/developers
   developers/doxygen
   developers/gnumake
   developers/how_to_guides
   developers/faq
.. good to have in the future:
..   developers/repostructure

Maintenance
-----------
.. toctree::
   :caption: MAINTENANCE
   :maxdepth: 1
   :hidden:

   maintenance/release

Epilogue
--------
.. toctree::
   :caption: EPILOGUE
   :maxdepth: 1
   :hidden:

   glossary
   governance
   acknowledgements
