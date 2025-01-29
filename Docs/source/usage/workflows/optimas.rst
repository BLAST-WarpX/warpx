.. _optimas:

Optimizing with Optimas
========================

`optimas <https://github.com/optimas-org/optimas>`__ is an open-source Python library that enables highly scalable parallel optimization, from a typical laptop to exascale HPC systems.
While a WarpX simulation can provide insight in some physics, it remains a single point evaluation in the space of parameters.
If you have a simulation ready for use, but would like to (i) scan over some input parameters uniformly for, e.g., a tolerance study, or (ii) have a random evaluation of the space of input parameters within a given span or (iii) tune some input parameters to optimize an output parameter, e.g., beam emittance, energy spread, etc., optimas provides these capabilities and will take care of tasks monitoring with fault tolerance on multiple platforms (optimas targets modern HPC platforms like Perlmutter and Frontier).

A more detailed description of optimas is provided in the `optimas documentation <https://optimas.readthedocs.io/en/latest/>`__.
In particular, the online optimas documentation provides `an example optimization with optimas that runs WarpX simulations <https://optimas.readthedocs.io/en/latest/examples/bo_with_warpx.html>`__.
