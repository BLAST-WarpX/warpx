---
name: warpx-answer-user-question
description: Draft a response to a WarpX user question (from a GitHub issue, discussion, or email).
---

# Answer WarpX User Question

Draft a helpful, accurate response to a WarpX user question, in the style of an experienced WarpX developer.

## Step 1 — Gather the question

If a GitHub issue or discussion URL was provided, fetch it with `gh`. If a plain-text question was pasted, use it directly.

Identify:
- **What the user is trying to do** (their end goal, not just their literal question)
- **What they have tried** (input file snippets, error messages, partial results)
- **Their apparent expertise level** (new to WarpX? PIC codes in general? HPC?)
- **What information is missing** that would be needed to give a complete answer

## Step 2 — Categorize the question

Assign the question to one or more of these categories:

| Category | Typical signals |
|---|---|
| **A. Installation / Build** | CMake errors, missing libraries, conda/spack, CUDA compilation, module conflicts |
| **B. Input File / Parameters** | Wrong parameter name or value, PICMI vs native syntax, laser setup, particle injection, boundary conditions |
| **C. Diagnostics & Output** | Output format, reduced diagnostics syntax, checkpoint/restart, post-processing |
| **D. Physics / Numerical** | Numerical heating, energy non-conservation, CFL, boosted frame artifacts, resolution |
| **E. Feature Availability** | "Does WarpX support X?" |
| **F. Performance & HPC** | Slow I/O, OOM, multi-node MPI, GPU choice |
| **G. Python / PICMI** | `pywarpx` API, callbacks, `add_particles`, PICMI vs native mapping |
| **H. Bug Report** | Reproducible unexpected behavior, wrong results, crashes |

## Step 3 — Search for relevant information

Before drafting any answer, actively search. The answer must be grounded in WarpX code, docs, and prior issues — not generated from general knowledge.

Search these sources as relevant to the category:

**Documentation** (`Docs/source/`):
- `usage/parameters.rst` — all input parameters (the primary reference)
- `usage/python.rst` — PICMI/Python interface
- `install/cmake.rst` — build options
- `dataanalysis/` — output formats, openPMD, post-processing tools

**Source code** — key directories:
- `Source/Laser/`, `Source/Particles/`, `Source/FieldSolver/`, `Source/Diagnostics/`, `Source/BoundaryConditions/`, `Source/Python/`
- For parameter reading: search for `pp.query`/`pp.get` with the parameter name
- For reduced diagnostics: `Source/Diagnostics/ReducedDiags/`

**Examples** (`Examples/Tests/`, `Examples/Physics_applications/`):
- Search for examples that demonstrate the relevant feature

**Past GitHub issues and discussions**:
- Search issues and discussions for similar questions and their resolutions
- For feature availability: also search open issues/PRs for planned work

## Step 4 — Assess whether more information is needed

Decide: **do you have enough information to give a complete, accurate answer?**

If not, identify the 1-3 most important missing pieces. Common things to ask for:
- Full input file (or the relevant section)
- Full error message or stack trace
- WarpX version, platform, CMake configuration (for build issues)
- Minimal reproducible script (for Python/PICMI issues)

## Step 5 — Draft the response

**Tone** — match the style of experienced WarpX developers:
- Warm and welcoming: open with "Hi @username," or "Thanks for your question!"
- Patient and non-condescending
- Direct: give the answer first, then the explanation
- Concrete: include corrected input file snippets or code examples inline
- Honest about limitations: if a feature doesn't exist, say so clearly

**Structure by situation:**

**Answerable question:** (1) Direct answer upfront, (2) explanation if needed, (3) corrected snippet or example, (4) link to relevant docs page on `warpx.readthedocs.io` or example file in repo

**Bug report:** (1) Confirm the bug with explanation, (2) workaround if one exists, (3) state if a fix PR is already open, (4) ask user to verify once merged.

**Missing info:** (1) Explain what's needed and why, (2) provide partial answer if possible.

**Feature not available:** (1) Confirm clearly, (2) explain workarounds (Python callbacks, post-processing, etc.), (3) link to tracking issue/PR if one exists, (4) invite contribution.

**Common gotchas to check proactively:**
- Wrong sign on `q_tot` for electrons (must be negative; positive silently causes `BeamRelevant` to output zeros)
- CFL too low for EM laser simulations (should be ~0.999, not 0.9)
- Missing `<diag_name>.intervals` on reduced diagnostics (nothing is written)
- HDF5 I/O slow on HPC filesystems — recommend ADIOS2 BP (`openpmd_backend = bp`)
- `get_particle_uz()` returns a list of per-box arrays in Python, not a flat array
- PICMI `focal_position` (absolute position) vs native `profile_focal_distance` (distance from antenna) — different sign conventions
- `do_continuous_injection` is for moving-window plasma injection only, not for fixed-domain plasma sources
- `particle_fields` parser expressions cannot cross-reference other computed moments

## Step 6 — Present the draft

Show the draft and ask:

> Does this response look good? Would you like me to adjust anything before you post it?

Only post the reply to GitHub if asked explicitly. Always show the draft first.
