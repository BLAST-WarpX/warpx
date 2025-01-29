Job Submission
''''''''''''''

* ``qsub your_job_script.qsub``


Job Control
'''''''''''

* interactive job:

  * ``qsub -I``

* details for my jobs:

  * ``qstat -f 12345`` all details for job with <job id> ``12345``
  * ``qstat -u $(whoami)`` all jobs under my user name

* details for queues:

  * ``qstat -a queueName`` show all jobs in a queue
  * ``pbs_free -l`` compact view on free and busy nodes
  * ``pbsnodes`` list all nodes and their detailed state (free, busy/job-exclusive, offline)

* communicate with job:

  * ``qdel <job id>`` abort job
  * ``qsig -s <signal number> <job id>`` send signal or signal name to job
  * ``qalter -lwalltime=12:00:00 <job id>`` change the walltime of a job
  * ``qalter -Wdepend=afterany:54321 12345`` only start job ``12345`` after job with id ``54321`` has finished
  * ``qhold <job id>`` prevent the job from starting
  * ``qrls <job id>`` release the job to be eligible for run (after it was set on hold)


References
''''''''''

* https://www.openpbs.org
