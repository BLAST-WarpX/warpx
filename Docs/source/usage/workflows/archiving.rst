.. _archiving:

Archiving
=========

Archiving simulation inputs, scripts and output data is a common need for computational physicists.
Here are some popular tools and workflows to make archiving easy.


.. _archiving-hpss:

HPC Systems: HPSS
-----------------

A very common tape filesystem is HPSS, e.g., on `NERSC <https://docs.nersc.gov/filesystems/archive/>`__ or `OLCF <https://docs.olcf.ornl.gov/data/index.html#data-storage-and-transfers>`__.

* What's in my archive file system? ``hsi ls``
* Already something in my archive location? ``hsi ls 2019/cool_campaign/`` as usual
* Let's create a neat **directory structure**:

  * new directory on the archive: ``hsi mkdir 2021``
  * create sub-dirs per campaign as usual: ``hsi mkdir 2021/reproduce_paper``
* **Create** an archive of a simulation: ``htar -cvf 2021/reproduce_paper/sim_042.tar /global/cfs/cdirs/m1234/ahuebl/reproduce_paper/sim_042``

  * This *copies* all files over to the tape filesystem and stores them as a single ``.tar`` archive
  * The first argument here will be the new archive ``.tar`` file on the archive file system, all following arguments (can be multiple, separated by a space) are locations to directories and files on the parallel file system.
  * Don't be confused, these tools also create an index ``.tar.idx`` file along it; just leave that file be and don't interact with it
* **Change permissions** of your archive, so your team can read your files:

  * Check the unix permissions via ``hsi ls -al 2021/`` and ``hsi ls -al 2021/reproduce_paper/``
  * *Files* must be group (g) readable (r): ``hsi chmod g+r 2021/reproduce_paper/sim_042.tar``
  * *Directories* must be group (g) readable (r) and group accessible (x): ``hsi chmod -R g+rx 2021``
* **Restore** things:

  * ``mkdir here_we_restore``
  * ``cd here_we_restore``
  * ``htar -xvf 2021/reproduce_paper/sim_42.tar``

    * this *copies* the ``.tar`` file back from tape to our parallel filesystem and extracts its content in the current directory

Argument meaning: ``-c`` create; ``-x`` extract; ``-v`` verbose; ``-f`` tar filename.
That's it, folks!

.. note::

   Sometimes, for large dirs, ``htar`` takes a while.
   You could then consider running it as part of a (single-node/single-cpu) job script.


.. _archiving-desktop:

Desktops/Laptops: Cloud Drives
------------------------------

Even for small simulation runs, it is worth to create data archives.
A good location for such an archive might be the cloud storage provided by one's institution.

Tools like `rclone <https://rclone.org>`__ can help with this, e.g., to quickly sync a large amount of directories to a Google Drive.


.. _archiving-globus:

Asynchronous File Copies: Globus
--------------------------------

The scientific data service `Globus <https://app.globus.org>`__ allows to perform large-scale data copies, between HPC centers as well as local computers, with ease and a graphical user interface.
Copies can be kicked off asynchronously, often use dedicated internet backbones and are checked when transfers are complete.

Many HPC centers also add their archives as a storage endpoint and one can download a client program to add also one's desktop/laptop.


.. _archiving-open-data:

Scientific Data for Publications
--------------------------------

It is good practice to make computational results accessible, scrutinizable and ideally even reusable.

For data artifacts up to approximately 50 GB, consider using free services like `Zenodo <https://www.zenodo.org>`__ and `Figshare <https://figshare.com>`__ to store supplementary materials of your publications.

For more information, see the open science movement, open data and open access.

.. note::

   More information, guidance and templates will be posted here in the future.
