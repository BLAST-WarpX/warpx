.. _development-python:

Processing PICMI Input Options
==============================

The input parameters in a WarpX PICMI file are processed in two layers.
The first layer is the Python level API, which mirrors the :ref:`C++ application input structure <running-cpp-parameters>`; the second is the translation from the PICMI input to the equivalent :ref:`app (AMReX) input file parameters <running-cpp-parameters>`.

The two layers are described below.

Input parameters
----------------

In a C++ input file, each of the parameters has a prefix, for example ``geometry`` in ``geometry.prob_lo``.
For each of these prefixes, an instance of a Python class is created and the parameters saved as attributes.
This construction is used since the lines in the input file look very much like a Python assignment statement,
assigning attributes of class instances, for example ``geometry.dims = 3``.

Many of the prefix instances are predefined, for instance ``geometry`` is created in the file ``Python/pywarpx/Geometry.py``.
In that case, ``geometry`` is an instance of the class ``Bucket`` (specified in ``Python/pywarpx/Bucket.py``),
the general class for prefixes.
It is called ``Bucket`` since its main purpose is a place to hold attributes.
Most of the instances are instances of the ``Bucket`` class.
There are exceptions, such as ``constants`` and ``diagnostics`` where extra processing is needed.

There can also be instances created as needed.
For example, for the particle species, an instance is created for each species listed in ``particles.species_names``.
This gives a place to hold the parameters for the species, e.g., ``electrons.mass``.

The instances are then used to generate the input parameters.
Each instance can generate a list of strings, one for each attribute.
This happens in the ``Bucket.attrlist`` method.
The strings will be the lines as in an input file, for example ``"electrons.mass = m_e"``.
The lists for each instance are gathered into one long list in the ``warpx`` instance (of the class ``WarpX`` defined in
``Python/pywarpx/WarpX.py``).
This instance has access to all of the predefined instances as well as lists of the generated instances.

In both of the ways that WarpX can be run with Python, that list of input parameter strings will be generated.
This is done in the routine ``WarpX.create_argv_list`` in ``Python/pywarpx/WarpX.py``.
If WarpX will be run directly in Python, that list will be sent to the ``amrex_init`` routine as the ``argv``.
This is as if all of the input parameters had been specified on the command line.
If Python is only used as a prepocessor to generate the input file, the list are the strings that are written out to create the
input file.

There are two input parameters that do not have prefixes, ``max_step`` and ``stop_time``.
These are handled via keyword arguments in the ``WarpX.create_argv_list`` method.

Conversion from PICMI
---------------------

In the PICMI implementation, defined in ``Python/pywarpx/picmi.py``, for each PICMI class, a class was written that
inherits the PICMI class and does the processing of the input.
Each of the WarpX classes has two methods, ``init`` and ``initialize_inputs``.
The ``init`` method is called during the creation of the class instances that happens in the user's PICMI input file.
This is part of the standard - each of the PICMI classes call the method ``handle_init`` from the constructor ``__init__`` routines.
The main purpose is to process application specific keyword arguments (those that start with ``warpx_`` for example).
These are then passed into the ``init`` methods.
In the WarpX implementation, in the ``init``, each of the WarpX specific arguments are saved as attributes of the implementation
class instances.

It is in the second method, ``initialize_inputs``, where the PICMI input parameters are translated into WarpX input parameters.
This method is called later during the initialization.
The prefix instances described above are all accessible in the implementation classes (via the ``pywarpx`` module).
For each PICMI input quantity, the appropriate WarpX input parameters are set in the prefix classes.
As needed, for example in the ``Species`` class, the dynamic prefix instances are created and the attributes set.

Simulation class
----------------

The ``Simulation`` class ties it all together.
In a PICMI input file, all information is passed into the ``Simulation`` class instance, either through the constructor
or through ``add_`` methods.
Its ``initialize_inputs`` routine initializes the input parameters it handles and also calls the ``initialize_inputs``
methods of all of the PICMI class instances that have been passed in, such as the field solver, the particles species,
and the diagnostics.
As with other PICMI classes, the ``init`` routine is called by the constructor and ``initialize_inputs`` is called during
initialization.
The initialization happens when either the ``write_input_file`` method is called or the ``step`` method.
After ``initialize_inputs`` is finished, the attributes of the prefix instances have been filled in, and the process described
above happens, where the prefix instances are looped over to generate the list of input parameter strings (that is either written
out to a file or passed in as ``argv``).
The two parameters that do not have a prefix, ``max_step`` and ``stop_time``, are passed into the ``warpx`` method as keyword
arguments.
