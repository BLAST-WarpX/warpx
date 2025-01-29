.. _plot-timestep-duration:

Plot timestep duration
======================
We provide a simple python script to generate plots of the timestep duration
from the stdandard output of WarpX (provided that ``warpx.verbose`` is set to 1):
:download:`plot_timestep_duration.py <../../../../Tools/PostProcessing/plot_timestep_duration.py>`.

If the standard output of a simulation has been redirected to a file named ``log_file``,
the script can be used as follows:

::

    python plot_timestep_duration.py log_file

The script generates two pictures: ``log_file_ts_duration.png``, which shows the duration
of each timestep in seconds as a function of the timestep number, and ``log_file_ts_cumulative_duration.png``,
which shows the total duration of the simulation as a function of the timestep number.
