.. _developers-warning-logger:

Warning logger
==============

The  ⚠️ warning logger ⚠️ allows grouping the warning messages raised during the
simulation, in order to display them together in a list
(e.g., right after step 1 and at the end of the simulation).

General description
-------------------

If no warning messages are raised, the warning list should look as follows:

.. code-block:: sh

   **** WARNINGS ******************************************************************
   * GLOBAL warning list  after  [ FIRST STEP ]
   *
   * No recorded warnings.
   ********************************************************************************

On the contrary, if warning messages are raised, the list should look as follows:

.. code-block:: sh

   **** WARNINGS ******************************************************************
   * GLOBAL warning list  after  [ FIRST STEP ]
   *
   * --> [!! ] [Species] [raised once]
   *     Both 'electrons.charge' and electrons.species_type' are specified.
   *     electrons.charge' will take precedence.
   *     @ Raised by: ALL
   *
   * --> [!! ] [Species] [raised once]
   *     Both 'electrons.mass' and electrons.species_type' are specified.
   *     electrons.mass' will take precedence.
   *     @ Raised by: ALL
   *
   ********************************************************************************

Here, ``GLOBAL`` indicates that warning messages are gathered across all the MPI ranks (specifically
after the ``FIRST STEP``).

Each entry of warning list respects the following format:

.. code-block:: sh

   * --> [PRIORITY] [TOPIC] [raised COUNTER]
   *     MULTILINE MESSAGE
   *     MULTILINE MESSAGE
   *     @ Raised by: WHICH_RANKS

where:

* ``[PRIORITY]`` can be ``[!  ]`` (low priority), ``[!! ]`` (medium priority) or ``[!!!]`` (high priority). It indicates the importance of the warning.
* ``[TOPIC]`` indicates which part of the code is concerned by the warning (e.g., particles, laser, parallelization...)
* ``MULTILINE MESSAGE`` is an arbitrary text message. It can span multiple-lines. Text is wrapped automatically.
* ``COUNTER`` indicates the number of times the warning was raised **across all the MPI ranks**. This means that if we run WarpX with 2048 MPI ranks and each rank raises the same warning once, the displayed message will be ``[raised 2048 times]``. Possible values are ``once``, ``twice``, ``XX times``
* ``WHICH_RANKS`` can be either ``ALL`` or a sequence of rank IDs. It is the list of the MPI ranks which have raised the warning message.

Entries are sorted first by priority (high priority first), then by topic (alphabetically) and finally by text message (alphabetically).

How to record a warning for later display
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the code, instead of using ``amrex::Warning`` to immediately print a warning message, the following method should be called:

.. code-block:: cpp

   ablastr::warn_manager::WMRecordWarning(
      "QED",
      "Using default value (2*me*c^2) for photon energy creation threshold",
      ablastr::warn_manager::WarnPriority::low);

In this example, ``QED`` is the topic, ``Using [...]`` is the warning message and ``ablastr::warn_manager::WarnPriority::low`` is the priority.
`RecordWarning` is **not** a collective call and should also be thread-safe (it can be called in OpenMP loops).
In case the user wants to also print the warning messages immediately, the runtime parameter ``warpx.always_warn_immediately`` can be set to ``1``.
The Warning manager is a singleton class defined in ``Source/ablastr/warn_manager/WarnManager.H```

How to print the warning list
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The warning list can be printed as follows:

.. code-block:: cpp

   amrex::Print() << ablastr::warn_manager::GetWMInstance().PrintGlobalWarnings("THE END");

where the string is a temporal marker that appears in the warning list.
At the moment this is done right after step one and at the end of the simulation.
Calling this method triggers several collective calls that allow merging all the warnings recorded by all the MPI ranks.

Implementation details
----------------------

How warning messages are recorded
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Warning messages are stored by each rank as a map associating each
message with a counter.
A message is defined by its priority, its topic and its text.
Given two messages, if any of these components differ between the
two, the messages are considered as different.

How the global warning list is generated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to generate the global warning list we follow the strategy outlined below.

1. Each MPI rank has a ``map<Msg, counter>``, associating each with a counter, which counts how many times the warning has been raised on that rank.
2. When ``PrintGlobalWarnings`` is called, the MPI ranks send to the I/O rank the number of different warnings that they have observed. The I/O rank finds the rank having more warnings and broadcasts 📢 this information back to all the others. This rank, referred in the following as *gather rank*, will lead  👑 the generation of the global warning list
3. The *gather rank* serializes its warning messages [📝,📝,📝,📝,📝...] into a byte array 📦 and  broadcasts 📢 this array to all the other ranks.
4. The other ranks unpack this byte array 📦, obtaining a list of messages [📝,📝,📝,📝,📝...]
5. For each message seen by the *gather rank* , each rank prepares a vector containing the number of times it has seen that message (i.e., the counter in ``map<Msg, counter>`` if ``Msg`` is in the map): [1️⃣,0️⃣,1️⃣,4️⃣,0️⃣...]
6. In addition, each rank prepares a vector containing the messages seen only by that rank, associated with the corresponding counter: [(📝,1️⃣), (📝,4️⃣),...]
7. Each rank appends the second list to the first one and packs them into a byte array: [1️⃣,0️⃣,1️⃣,4️⃣,0️⃣...] [(📝,1️⃣), (📝,4️⃣),...] --> 📦
8. Each rank sends 📨 this byte array to the *gather rank*, which puts them together in a large byte vector [📦,📦,📦,📦,📦...]
9. The *gather rank* parses the byte array, adding the counters of the other ranks to its counters, adding new messages to the message list, and keeping track of which rank has generated which warning 📜
10. If the *gather rank* is also the I/O rank, then we are done  🎉, since the rank has a list of messages, global counters and ranks lists  [(📝,4️⃣,📜 ), (📝,1️⃣,📜 ),... ]
11. If the *gather rank* is **not** the I/O rank, then it packs the list into a byte array and sends  📨 it to the I/O rank, which unpacks it: *gather rank* [(📝,4️⃣,📜 ), (📝,1️⃣,📜 ),... ] --> 📦 --> 📨 --> 📦 --> [(📝,4️⃣,📜 ), (📝,1️⃣,📜 ),... ] I/O rank

This procedure is described in more details in these `slides <https://drive.google.com/file/d/1f7w-iCGWwRk4OR_Hu_hPzWJYvWrfj6U8/view?usp=sharing>`_.

How to test the warning logger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to test the warning logger there is the possibility to inject "artificial" warnings with the inputfile.
For instance, the following inputfile

.. code-block:: sh

   #################################
   ####### GENERAL PARAMETERS ######
   #################################
   max_step = 10
   amr.n_cell =  128 128
   amr.max_grid_size = 64
   amr.blocking_factor = 32
   amr.max_level = 0
   geometry.dims = 2
   geometry.prob_lo     = -20.e-6   -20.e-6    # physical domain
   geometry.prob_hi     =  20.e-6    20.e-6

   #################################
   ####### Boundary condition ######
   #################################
   boundary.field_lo = periodic periodic
   boundary.field_hi = periodic periodic

   #################################
   ############ NUMERICS ###########
   #################################
   warpx.serialize_initial_conditions = 1
   warpx.verbose = 1
   warpx.cfl = 1.0
   warpx.use_filter = 0

   # Order of particle shape factors
   algo.particle_shape = 1

   #################################
   ######## DEBUG WARNINGS #########
   #################################

   warpx.test_warnings = w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22

   w1.topic    = "Priority Sort Test"
   w1.msg      = "Test that priority is correctly sorted"
   w1.priority = "low"
   w1.all_involved = 1

   w2.topic    = "Priority Sort Test"
   w2.msg        = "Test that priority is correctly sorted"
   w2.priority = "medium"
   w2.all_involved = 1

   w3.topic    = "Priority Sort Test"
   w3.msg      = "Test that priority is correctly sorted"
   w3.priority = "high"
   w3.all_involved = 1

   w4.topic    = "ZZA Topic sort Test"
   w4.msg      = "Test that topic is correctly sorted"
   w4.priority = "medium"
   w4.all_involved = 1

   w5.topic    = "ZZB Topic sort Test"
   w5.msg      = "Test that topic is correctly sorted"
   w5.priority = "medium"
   w5.all_involved = 1

   w6.topic    = "ZZC Topic sort Test"
   w6.msg      = "Test that topic is correctly sorted"
   w6.priority = "medium"
   w6.all_involved = 1

   w7.topic    = "Msg sort Test"
   w7.msg      = "AAA Test that msg is correctly sorted"
   w7.priority = "medium"
   w7.all_involved = 1

   w8.topic    = "Msg sort Test"
   w8.msg      = "BBB Test that msg is correctly sorted"
   w8.priority = "medium"
   w8.all_involved = 1

   w9.topic    = "Long line"
   w9.msg      = "Test very long line: a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a"
   w9.priority = "medium"
   w9.all_involved = 1

   w10.topic    = "Repeated warnings"
   w10.msg      = "Test repeated warnings"
   w10.priority = "high"
   w10.all_involved = 1

   w11.topic    = "Repeated warnings"
   w11.msg      = "Test repeated warnings"
   w11.priority = "high"
   w11.all_involved = 1

   w12.topic    = "Repeated warnings"
   w12.msg      = "Test repeated warnings"
   w12.priority = "high"
   w12.all_involved = 1

   w13.topic    = "Not all involved (0)"
   w13.msg      = "Test warnings raised by a fraction of ranks"
   w13.priority = "high"
   w13.all_involved = 0
   w13.who_involved = 0

   w14.topic    = "Not all involved (0)"
   w14.msg      = "Test warnings raised by a fraction of ranks"
   w14.priority = "high"
   w14.all_involved = 0
   w14.who_involved = 0

   w15.topic    = "Not all involved (1)"
   w15.msg      = "Test warnings raised by a fraction of ranks"
   w15.priority = "high"
   w15.all_involved = 0
   w15.who_involved = 1

   w16.topic    = "Not all involved (1,2)"
   w16.msg      = "Test warnings raised by a fraction of ranks"
   w16.priority = "high"
   w16.all_involved = 0
   w16.who_involved = 1 2

   w17.topic    = "Different counters"
   w17.msg      = "Test that different counters are correctly summed"
   w17.priority = "low"
   w17.all_involved = 1

   w18.topic    = "Different counters"
   w18.msg      = "Test that different counters are correctly summed"
   w18.priority = "low"
   w18.all_involved = 1

   w19.topic    = "Different counters"
   w19.msg      = "Test that different counters are correctly summed"
   w19.priority = "low"
   w19.all_involved = 0
   w19.who_involved = 0

   w20.topic    = "Different counters B"
   w20.msg      = "Test that different counters are correctly summed"
   w20.priority = "low"
   w20.all_involved = 1

   w21.topic    = "Different counters B"
   w21.msg      = "Test that different counters are correctly summed"
   w21.priority = "low"
   w21.all_involved = 1

   w22.topic    = "Different counters B"
   w22.msg      = "Test that different counters are correctly summed"
   w22.priority = "low"
   w22.all_involved = 0
   w22.who_involved = 1

should generate the following warning list (if run on 4 MPI ranks):

.. code-block:: sh

   **** WARNINGS ******************************************************************
   * GLOBAL warning list  after  [ THE END ]
   *
   * --> [!!!] [Not all involved (0)] [raised twice]
   *     Test warnings raised by a fraction of ranks
   *     @ Raised by: 0
   *
   * --> [!!!] [Not all involved (1)] [raised once]
   *     Test warnings raised by a fraction of ranks
   *     @ Raised by: 1
   *
   * --> [!!!] [Not all involved (1,2)] [raised twice]
   *     Test warnings raised by a fraction of ranks
   *     @ Raised by: 1 2
   *
   * --> [!!!] [Priority Sort Test] [raised 4 times]
   *     Test that priority is correctly sorted
   *     @ Raised by: ALL
   *
   * --> [!!!] [Repeated warnings] [raised 12 times]
   *     Test repeated warnings
   *     @ Raised by: ALL
   *
   * --> [!! ] [Long line] [raised 4 times]
   *     Test very long line: a a a a a a a a a a a a a a a a a a a a a a a a a a a
   *     a a a a a a a a a a a a a a a a a a a a a a a a a a a a a
   *     @ Raised by: ALL
   *
   * --> [!! ] [Msg sort Test] [raised 4 times]
   *     AAA Test that msg is correctly sorted
   *     @ Raised by: ALL
   *
   * --> [!! ] [Msg sort Test] [raised 4 times]
   *     BBB Test that msg is correctly sorted
   *     @ Raised by: ALL
   *
   * --> [!! ] [Priority Sort Test] [raised 4 times]
   *     Test that priority is correctly sorted
   *     @ Raised by: ALL
   *
   * --> [!! ] [ZZA Topic sort Test] [raised 4 times]
   *     Test that topic is correctly sorted
   *     @ Raised by: ALL
   *
   * --> [!! ] [ZZB Topic sort Test] [raised 4 times]
   *     Test that topic is correctly sorted
   *     @ Raised by: ALL
   *
   * --> [!! ] [ZZC Topic sort Test] [raised 4 times]
   *     Test that topic is correctly sorted
   *     @ Raised by: ALL
   *
   * --> [!  ] [Different counters] [raised 9 times]
   *     Test that different counters are correctly summed
   *     @ Raised by: ALL
   *
   * --> [!  ] [Different counters B] [raised 9 times]
   *     Test that different counters are correctly summed
   *     @ Raised by: ALL
   *
   * --> [!  ] [Priority Sort Test] [raised 4 times]
   *     Test that priority is correctly sorted
   *     @ Raised by: ALL
   *
   ********************************************************************************
