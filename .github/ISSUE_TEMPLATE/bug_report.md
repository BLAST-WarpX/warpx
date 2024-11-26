---
name: Bug report
about: Report a bug or unexpected behavior.
labels: [bug]
---

_Please remove any sensitive information (e.g., passwords, API keys) from your submission.
Please check the relevant boxes and fill in the specific versions or details for the relevant items.
Thank you for taking the time to report this issue. We will respond as soon as possible._

## Description
A clear and concise description of the bug.

## Expected behavior
What did you expect to happen when you encountered the issue?

## How to reproduce
Please provide (if available):
- WarpX inputs files
- PICMI Python files
- Python post-processing scripts

If you are unable to provide certain files or scripts, please describe the steps you took to encounter the issue.

Please minimize your inputs/scripts to be concise and focused on the issue.
For instance, make the simulation scripts as small and fast to run as possible.

## System information
Please check all relevant boxes and provide details.

- Operating system (name and version):
  - [ ] Linux: e.g., Ubuntu 22.04 LTS
  - [ ] macOS: e.g., macOS Monterey 12.4
  - [ ] Windows: e.g., Windows 11 Pro
- Version of WarpX: e.g., latest, 24.10, etc.
- Installation method:
  - [ ] Conda
  - [ ] Spack
  - [ ] PyPI
  - [ ] Brew
  - [ ] From source with CMake
  - [ ] Module system on an HPC cluster
- Other dependencies: yes/no, describe
- Computational resources:
  - [ ] MPI: e.g., 2 MPI processes
  - [ ] OpenMP: e.g., 2 OpenMP threads
  - [ ] CPU: e.g., 2 CPUs
  - [ ] GPU: e.g., 2 GPUs (NVIDIA, AMD, etc.)

If you encountered the issue on an HPC cluster, please check our [HPC documentation](https://warpx.readthedocs.io/en/latest/install/hpc.html) to see if your HPC cluster is already supported.

## Steps taken so far
What troubleshooting steps have you taken so far, and what were the results?

Have you tried debugging the code, following the instructions in our [debugging documentation](https://warpx.readthedocs.io/en/latest/usage/workflows/debugging.html)?

## Additional information
If applicable, please add any additional information that may help explain the issue, such as log files (e.g., build logs, error logs, etc.), error messages (e.g., compiler errors, runtime errors, etc.), screenshots, or other relevant details.
