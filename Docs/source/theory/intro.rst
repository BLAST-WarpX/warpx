.. _theory:

Introduction
============

.. figure:: Plasma_acceleration_sim.png
   :alt: Plasma laser-driven (top) and charged-particles-driven (bottom) acceleration (rendering from 3-D Particle-In-Cell simulations). A laser beam (red and blue disks in top picture) or a charged particle beam (red dots in bottom picture) propagating (from left to right) through an under-dense plasma (not represented) displaces electrons, creating a plasma wakefield that supports very high electric fields (pale blue and yellow). These electric fields, which can be orders of magnitude larger than with conventional techniques, can be used to accelerate a short charged particle beam (white) to high-energy over a very short distance.

   Plasma laser-driven (top) and charged-particles-driven (bottom) acceleration (rendering from 3-D Particle-In-Cell simulations). A laser beam (red and blue disks in top picture) or a charged particle beam (red dots in bottom picture) propagating (from left to right) through an under-dense plasma (not represented) displaces electrons, creating a plasma wakefield that supports very high electric fields (pale blue and yellow). These electric fields, which can be orders of magnitude larger than with conventional techniques, can be used to accelerate a short charged particle beam (white) to high-energy over a very short distance.

Computer simulations have had a profound impact on the design and understanding of past and present plasma acceleration experiments :cite:p:`i-Tsungpop06,i-Geddesjp08,i-Geddesscidac09,i-Geddespac09,i-Huangscidac09`. Accurate modeling of wake formation, electron self-trapping and acceleration require fully kinetic methods (usually Particle-In-Cell) using large computational resources due to the wide range of space and time scales involved. Numerical modeling complements and guides the design and analysis of advanced accelerators, and can reduce development costs significantly. Despite the major recent experimental successes :cite:p:`i-LeemansPRL2014,i-Blumenfeld2007,i-BulanovSV2014,i-Steinke2016`, the various advanced acceleration concepts need significant progress to fulfill their potential. To this end, large-scale simulations will continue to be a key component toward reaching a detailed understanding of the complex interrelated physics phenomena at play.

For such simulations,
the most popular algorithm is the Particle-In-Cell (or PIC) technique,
which represents electromagnetic fields on a grid and particles by
a sample of macroparticles.
However, these simulations are extremely computationally intensive, due to the need to resolve the evolution of a driver (laser or particle beam) and an accelerated beam into a structure that is orders of magnitude longer and wider than the accelerated beam.
Various techniques or reduced models have been developed to allow multidimensional simulations at manageable computational costs: quasistatic approximation :cite:p:`i-Sprangleprl90,i-Antonsenprl1992,i-Krallpre1993,i-Morapop1997,i-Quickpic`,
ponderomotive guiding center (PGC) models :cite:p:`i-Antonsenprl1992,i-Krallpre1993,i-Quickpic,i-Benedettiaac2010,i-Cowanjcp11`, simulation in an optimal Lorentz boosted frame :cite:p:`i-Vayprl07,i-Bruhwileraac08,i-Vayscidac09,i-Vaypac09,i-Martinspac09,i-VayAAC2010,i-Martinsnaturephysics10,i-Martinspop10,i-Martinscpc10,i-Vayjcp2011,i-VayPOPL2011,i-Vaypop2011,i-Yu2016`,
expanding the fields into a truncated series of azimuthal modes :cite:p:`i-godfrey1985iprop,i-LifschitzJCP2009,i-DavidsonJCP2015,i-Lehe2016,i-AndriyashPoP2016`, fluid approximation :cite:p:`i-Krallpre1993,i-Shadwickpop09,i-Benedettiaac2010` and scaled parameters :cite:p:`i-Cormieraac08,i-Geddespac09`.

.. bibliography::
    :keyprefix: i-
