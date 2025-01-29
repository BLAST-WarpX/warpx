.. _dataanalysis-workflows-tunneling:

Port Tunneling
==============

SSH port tunneling (port forwarding) is a secure way to access a computational service of a remote computer.
A typical workflow where you might need port tunneling is for Jupyter data analysis, e.g., when analyzing data on your desktop computer but working from your laptop.

Before getting started here, please note that many HPC centers offer a pre-installed Jupyter service, where tunnel is **not** needed.
For example, see the :ref:`NERSC Jupyter <post-processing-perlmutter>` and :ref:`OLCF Jupyter <post-processing-frontier>` services.


.. _dataanalysis-workflows-tunneling-background:

Introduction
------------

When running a service such as Jupyter from your command line, it will start a local (web) port.
The IPv4 address of your local computer is always ``127.0.0.1`` or the alias ``localhost``.

As a secure default, you cannot connect from outside your local computer to this port.
This prevents misconfigurations where one could, in the worst case, connect to your open port without authentication and execute commands with your user privileges.

One way to access your remote Jupyter desktop service from your laptop is to forward the port started remotely via an encrypted SSH connection to a local port on your current laptop.
The following section will explain the detailed workflow.


.. _dataanalysis-workflows-tunneling-workflow:

Workflow
--------

* you connect via SSH to your desktop at work, in a terminal (A) as usual

  * e.g., ssh ``username@your-computers-hostname.dhcp.lbl.gov``
  * start Jupyter locally in headless mode, e.g., ``jupyter lab --no-browser``
  * this will show you a ``127.0.0.1`` (aka ``localhost``) URL, by default on port TCP ``8888``
  * you cannot reach that URL, because you are not sitting on that computer, with your browser
* You now start a second terminal (B) locally, which forwards  the remote port 8888 to your local laptop

  * this step must be done **after** Jupyter was started on the desktop
  * ``ssh -L <laptop-port>:<Ip-as-seen-on-desktop>:<desktop-port> <desktop-ip> -N``
  * so concrete: ``ssh -L 8888:localhost:8888 your-computers-hostname.dhcp.lbl.gov -N``

    * note: Jupyter on the desktop will increase the port if already in use.
    * note: take another port on your laptop if you have local Jupyter instances still running
* Now open the browser on your local laptop, open the URL from Jupyter with ``.../127.0.0.1:8888/...`` in it

To close the connection down, do this:

  * stop Jupyter in terminal A: ``Ctrl+C`` and confirm with ``y``, ``Enter``
  * ``Ctrl+C`` the SSH tunnel in terminal B

.. figure:: https://user-images.githubusercontent.com/1353258/232120440-3965fa38-9ca6-4621-a100-2da74eb899cf.png
   :alt: Example view of remote started Jupyter service, active SSH tunnel, and local browser connecting to the service.
   :width: 100%
