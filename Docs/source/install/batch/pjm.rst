.. note::

   This section is a stub and improvements to complete the ``(TODO)`` sections are welcome.

Job Submission
''''''''''''''

* ``pjsub your_job_script.pjsub``


Job Control
'''''''''''

* interactive job:

  * ``pjsub --interact``

* details for my jobs:

  * ``pjstat`` status of all jobs
  * (TODO) all details for job with <job id> ``12345``
  * (TODO) all jobs under my user name

* details for queues:

  * (TODO) show all jobs in a queue
  * (TODO) compact view on free and busy nodes
  * (TODO) list all nodes and their detailed state (free, busy/job-exclusive, offline)

* communicate with job:

  * ``pjdel <job id>`` abort job
  * (TODO) send signal or signal name to job
  * (TODO) change the walltime of a job
  * (TODO) only start job ``12345`` after job with id ``54321`` has finished
  * ``pjhold <job id>`` prevent the job from starting
  * ``pjrls <job id>`` release the job to be eligible for run (after it was set on hold)


References
''''''''''

* https://www.bsc.es/user-support/arm.php#ToC-runningjobs
* https://www.cc.kyushu-u.ac.jp/scp/eng/system/ITO/02-2_batch.html
* https://www.r-ccs.riken.jp/en/fugaku/user-guide/
