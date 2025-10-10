Job Submission
''''''''''''''

* ``flux batch your_job_script.flux``


Job Control
'''''''''''

* `interactive job <https://flux-framework.readthedocs.io/projects/flux-core/en/latest/man1/flux-submit.html>`__:

  * ``flux submit --time-limit=1:00:00 --nodes=1 --tasks-per-node=4 --cores-per-task=8``

    * e.g. ``flux submit "hostname"``
  * GPU allocation requires additional flags, e.g. ``--gpus-per-task=1``

* details for my jobs:

  * ``flux jobs`` all jobs under my user name
  * ``flux job info abc123 jobspec`` all details for job with <job id> ``abc123``
  * ``flux job info 12345 eventlog`` history of events for job with <job id> ``12345``


* details for queues:

  * ``flux queue list`` list all queues
  * ``flux queue status`` show status of queues
  * *unclear/TODO* show start times for pending jobs
  * ``sinfo -p queueName`` show online/offline nodes in queue


* communicate with job:

  * ``flux cancel <job id>`` abort job
  * ``flux job kill --signal=<signal number> <job id>`` send signal or signal name to job
  * *unclear/TODO* change the walltime of a job
  * *unclear/TODO* only start job ``12345`` after job with id ``54321`` has finished
  * ``flux job urgency <job id> hold`` prevent the job from starting
  * ``flux job urgency <job id> default`` release the job to be eligible for run (after it was set on hold)


References
''''''''''

* `Flux commands <https://flux-framework.readthedocs.io/projects/flux-core/en/latest/man1/index.html>`__
