:orphan:

WarpX
-----

WarpX is an advanced **Particle-In-Cell** code.

It supports many features including:

    - Multiple types of field solvers (incl. for :ref:`Maxwell's equations <theory-em-pic>`, Poisson's equation, and :ref:`Ampere's law coupled with Ohm's law <theory-kinetic-fluid-hybrid-model>`)
    - Various grid geometries (1D/2D/3D Cartesian, cylindrical, spherical)
    - Multi-physics packages (incl. ionization, atomic, fusion and collisional physics, as well as quantum electrodynamics)
    - Advanced numerical methods (incl. explicit and implicit time advance, mesh refinement, boosted-frame simulations, embedded boundaries, pseudo-spectral solvers)

For details on these features, see the :ref:`theory section <theory>`.
WarpX has been applied to a wide variety of science projects, see :ref:`highlights <highlights>`.

In addition, WarpX is a *highly-parallel and highly-optimized code*:

    - Can run on multi-core CPUs as well as NVIDIA, AMD or Intel GPUs
    - Scales to the world's largest supercomputers and includes load balancing capabilities. WarpX was awarded the `2022 ACM Gordon Bell Prize <https://www.exascaleproject.org/ecp-supported-collaborative-teams-win-the-2022-acm-gordon-bell-prize-and-special-prize/>`__.
    - Multi-platform code that can run on Linux, macOS and Windows.
    - Can be run and :ref:`extended via its Python interface <usage-python-extend>`, e.g., to couple to other codes or AI/ML frameworks.

.. _contact:

Contact us
^^^^^^^^^^

The `WarpX GitHub repository <https://github.com/BLAST-WarpX/warpx>`__ is the main communication platform:

   - If you are new to WarpX or have a question, we encourage you to visit our `discussions page <https://github.com/BLAST-WarpX/warpx/discussions>`__ and connect with the community. This page is also a great place to browse answers to previously asked questions, post new ones, get help with installation, exchange ideas, and share feedback.
   - You can also explore the icons in the upper right corner of the `WarpX GitHub repository <https://github.com/BLAST-WarpX/warpx>`__ (e.g., ``Watch``, ``Star``, etc.): feel free to watch the repository if you want to receive updates, or to star the repository to support the project.
   - For bug reports, feature requests, or installation issues, you can also open a new `issue <https://github.com/BLAST-WarpX/warpx/issues>`__.

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
   usage/parameters
   usage/python
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
   dataanalysis/openpmd
   dataanalysis/yt
   dataanalysis/3dvisualizations
   dataanalysis/insitu
   dataanalysis/workflows

Theory
------
.. toctree::
   :caption: THEORY
   :maxdepth: 1
   :hidden:
   :titlesonly:

   theory/intro

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
