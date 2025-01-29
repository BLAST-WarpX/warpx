Job Submission
''''''''''''''

* ``sbatch your_job_script.sbatch``


Job Control
'''''''''''

* interactive job:

  * ``salloc --time=1:00:00 --nodes=1 --ntasks-per-node=4 --cpus-per-task=8``

    * e.g. ``srun "hostname"``
  * GPU allocation on most machines require additional flags, e.g. ``--gpus-per-task=1`` or ``--gres=...``

* details for my jobs:

  * ``scontrol -d show job 12345`` all details for job with <job id> ``12345``
  * ``squeue -u $(whoami) -l`` all jobs under my user name

* details for queues:

  * ``squeue -p queueName -l`` list full queue
  * ``squeue -p queueName --start`` (show start times for pending jobs)
  * ``squeue -p queueName -l -t R`` (only show running jobs in queue)
  * ``sinfo -p queueName`` (show online/offline nodes in queue)
  * ``sview`` (alternative on taurus: ``module load llview`` and ``llview``)
  * ``scontrol show partition queueName``

* communicate with job:

  * ``scancel <job id>`` abort job
  * ``scancel -s <signal number> <job id>`` send signal or signal name to job
  * ``scontrol update timelimit=4:00:00 jobid=12345`` change the walltime of a job
  * ``scontrol update jobid=12345 dependency=afterany:54321`` only start job ``12345`` after job with id ``54321`` has finished
  * ``scontrol hold <job id>`` prevent the job from starting
  * ``scontrol release <job id>`` release the job to be eligible for run (after it was set on hold)


References
''''''''''

* https://slurm.schedmd.com/documentation.html
