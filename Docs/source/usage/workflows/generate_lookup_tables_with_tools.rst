.. _generate-lookup-tables-with-tools:

Generate QED lookup tables using the standalone tool
====================================================

We provide tools to generate and convert into a human-readable format the QED lookup tables.
Such tools can be compiled with ``cmake`` by setting the flag ``WarpX_QED_TOOLS=ON`` (this
requires both ``PICSAR`` and ``Boost`` libraries). The tools are compiled alongside the WarpX executable
in the  folder ``bin``. We report here the help message displayed by the tools:

.. code-block:: console

    $ ./qed_table_reader -h
    ### QED Table Reader ###
    Command line options:
    -h [NO ARG] Prints all command line arguments
    -i [STRING] Name of the file to open
    --table [STRING] Either BW (Breit-Wheeler) or QS (Quantum Synchrotron)
    --mode [STRING] Precision of the calculations: either DP (double) or SP (single)
    -o [STRING] filename to save the lookup table in human-readable format

    $ ./qed_table_generator -h
    ### QED Table Generator ###
    Command line options:
    -h [NO ARG] Prints all command line arguments
    --table [STRING] Either BW (Breit-Wheeler) or QS (Quantum Synchrotron)
    --mode [STRING] Precision of the calculations: either DP (double) or SP (single)
    --dndt_chi_min [DOUBLE] minimum chi parameter for the dNdt table
    --dndt_chi_max [DOUBLE] maximum chi parameter for the dNdt table
    --dndt_how_many [INTEGR] number of points in the dNdt table
    --pair_chi_min [DOUBLE] minimum chi for the pair production table (BW only)
    --pair_chi_max [DOUBLE] maximum chi for the pair production table (BW only)
    --pair_chi_how_many [INTEGR] number of chi points in the pair production table (BW only)
    --pair_frac_how_many [INTEGR] number of frac points in the pair production table (BW only)
    --em_chi_min [DOUBLE] minimum chi for the photon emission table (QS only)
    --em_chi_max [DOUBLE] maximum chi for the photon emission production table (QS only)
    --em_frac_min [DOUBLE] minimum frac for the photon emission production table (QS only)
    --em_chi_how_many [INTEGR] number of chi points in the photon emission table (QS only)
    --em_frac_how_many [INTEGR] number of frac points in the photon emission table (QS only)
    -o [STRING] filename to save the lookup table

These tools are meant to be compatible with WarpX: ``qed_table_generator`` should generate
tables that can be loaded into WarpX and ``qed_table_reader`` should be able to read tables generated with WarpX.
It is not safe to use these tools to generate a table on a machine using a different endianness with respect to
the machine where the table is used.
