# WarpX

[![Code Status development](https://dev.azure.com/ECP-WarpX/WarpX/_apis/build/status/ECP-WarpX.WarpX?branchName=development)](https://dev.azure.com/ECP-WarpX/WarpX/_build/latest?definitionId=1&branchName=development)
[![Nightly Installation Tests](https://dev.azure.com/ECP-WarpX/WarpX/_apis/build/status/ECP-WarpX.Nightly?branchName=nightly&label=nightly%20packages)](https://dev.azure.com/ECP-WarpX/WarpX/_build?definitionId=2)
[![Documentation Status](https://readthedocs.org/projects/warpx/badge/?version=latest)](https://warpx.readthedocs.io)
[![Spack Version](https://img.shields.io/spack/v/warpx)](https://spack.readthedocs.io/en/latest/package_list.html#warpx)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/warpx)](https://anaconda.org/conda-forge/warpx)
[![Discussions](https://img.shields.io/badge/chat-discussions-turquoise.svg)](https://github.com/ECP-WarpX/WarpX/discussions)  
[![Supported Platforms](https://img.shields.io/badge/platforms-linux%20|%20osx%20|%20win-blue)](https://warpx.readthedocs.io/en/latest/install/users.html)
[![GitHub commits since last release](https://img.shields.io/github/commits-since/ECP-WarpX/WarpX/latest/development.svg)](https://github.com/ECP-WarpX/WarpX/compare/development)
[![HPSF](https://img.shields.io/badge/hosted%20by-HPSF-orange)](https://hpsf.io)
[![Language: C++17](https://img.shields.io/badge/language-C%2B%2B17-orange.svg)](https://isocpp.org/)
[![Language: Python](https://img.shields.io/badge/language-Python-orange.svg)](https://python.org/)  
[![License WarpX](https://img.shields.io/badge/license-BSD--3--Clause--LBNL-blue.svg)](https://spdx.org/licenses/BSD-3-Clause-LBNL.html)
[![DOI (source)](https://img.shields.io/badge/DOI%20(source)-10.5281/zenodo.4571577-blue.svg)](https://doi.org/10.5281/zenodo.4571577)
[![DOI (paper)](https://img.shields.io/badge/DOI%20(paper)-10.1109/SC41404.2022.00008-blue.svg)](https://doi.org/10.1109/SC41404.2022.00008)

## Overview

WarpX is an advanced **electromagnetic & electrostatic Particle-In-Cell** code.
It supports many features including Perfectly-Matched Layers (PML), mesh refinement, and the boosted-frame technique.

WarpX is a *highly-parallel and highly-optimized code*, which can run on GPUs and multi-core CPUs, and includes load balancing capabilities.
WarpX scales to the world's largest supercomputers and was awarded the [2022 ACM Gordon Bell Prize](https://www.exascaleproject.org/ecp-supported-collaborative-teams-win-the-2022-acm-gordon-bell-prize-and-special-prize/).

## Documentation

[![PICMI](https://img.shields.io/static/v1?label="works%20with"&message="PICMI"&color="blueviolet")](https://picmi-standard.github.io)
[![openPMD](https://img.shields.io/static/v1?label="works%20with"&message="openPMD"&color="blueviolet")](https://www.openPMD.org)
[![yt-project](https://img.shields.io/static/v1?label="works%20with"&message="yt"&color="blueviolet")](https://yt-project.org)

In order to learn how to install and run the code, please see the online documentation:
https://warpx.readthedocs.io

To contact the developers, feel free to open an issue on this repo, or visit our discussions page at https://github.com/ECP-WarpX/WarpX/discussions

## Contributing

[![AMReX](https://img.shields.io/static/v1?label="runs%20on"&message="AMReX"&color="blueviolet")](https://amrex-codes.github.io/)
[![PICSAR](https://img.shields.io/static/v1?label="runs%20on"&message="PICSAR"&color="blueviolet")](https://picsar.net)
[![openPMD-api](https://img.shields.io/static/v1?label="runs%20on"&message="openPMD-api"&color="blueviolet")](https://openpmd-api.readthedocs.io)
[![ADIOS](https://img.shields.io/static/v1?label="runs%20on"&message="ADIOS"&color="blueviolet")](https://csmd.ornl.gov/adios)
[![HDF5](https://img.shields.io/static/v1?label="runs%20on"&message="HDF5"&color="blueviolet")](https://www.hdfgroup.org/)
[![Ascent](https://img.shields.io/static/v1?label="runs%20on"&message="Ascent"&color="blueviolet")](http://www.ascent-dav.org)
[![SENSEI](https://img.shields.io/static/v1?label="runs%20on"&message="SENSEI"&color="blueviolet")](https://sensei-insitu.org)

Our workflow is described in [CONTRIBUTING.rst](CONTRIBUTING.rst).
We invite you to contribute to WarpX in any form following our [Code of Conduct](https://warpx.readthedocs.io/en/latest/coc.html), e.g., contribute to [discussions](https://github.com/ECP-WarpX/WarpX/discussions), help each other in [issues](https://github.com/ECP-WarpX/WarpX/issues), fix bugs, or add [documentation](https://warpx.readthedocs.io/en/latest/developers/documentation.html) and new functionality!

## Governance

WarpX is hosted by the High Performance Computing Foundation (HPSF).
If your organization wants to help steer the evolution of the HPC software ecosystem, visit [hpsf.io](https://hpsf.io) and consider joining!

The WarpX open governance model is described in [GOVERNANCE.rst](GOVERNANCE.rst).

## Copyright Notice

WarpX Copyright (c) 2018, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of any
required approvals from the U.S. Dept. of Energy).  All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Innovation & Partnerships Office at
IPO@lbl.gov.

Please see the full license agreement in [LICENSE.txt](LICENSE.txt).
Please see the notices in [NOTICE.txt](NOTICE.txt).
The SPDX license identifier is `BSD-3-Clause-LBNL`.
