.. _developers-llm:

LLM-Assisted WarpX Development
==============================

Large Language Models (LLMs) can assist with WarpX development tasks such as navigating the codebase, understanding existing implementations, writing new features, debugging, adding tests, and preparing pull requests.
This guide documents how the WarpX repository is configured for LLM-based coding assistants and how to get the most out of them.

.. note::

   LLM-generated code should always be reviewed carefully.
   LLMs can hallucinate APIs, miss physics constraints, or produce code that compiles but is incorrect.
   The configurations described below help by providing accurate, up-to-date context, but developer judgment remains essential.

   The WarpX community thus urges you to perform a careful, manual review of all LLM-generated code and documentation before asking for a review of your pull-request.
   This is important, otherwise you risk to waste the valuable time of our most proficient developers that will need to review your LLM-generated code.
   (Be considerate that WarpX developers can prompt an LLM just as efficiently as you can. Your critical thinking skill to make sense of the LLM-generated code and make it sensible for review and maintainable for the long term is what is needed!)

.. note::

   This section is not understood as an endorsement of any of the listed (or unlisted) coding assistants or MCP services.
   Contributions to this section documenting further services, clients, skills, etc. are encouraged.

.. tip::

   When working with LLM coding assistants, keep in mind that *"most best practices are based on one constraint: [the] context window fills up fast, and performance degrades as it fills"* (`Claude Code Best Practices <https://code.claude.com/docs/en/best-practices>`__).
   Keep instructions concise, use plan modes, break complex tasks into focused sessions, and provide targeted context rather than overwhelming the assistant with information.


AGENTS.md / CLAUDE.md
---------------------

The repository includes an ``AGENTS.md`` file at its root (as well as a ``CLAUDE.md``, which directly points to ``AGENTS.md``.)
These files are automatically loaded by LLM coding assistants (Claude Code reads ``CLAUDE.md``; other tools such as OpenAI Codex CLI read ``AGENTS.md``) to provide project-specific instructions.

The file contains in a compressed form instructions for an LLM agent:
With this file present, an LLM assistant working inside the WarpX repository will automatically know how to build, test, and style code without being told each time.

To update these instructions, edit ``AGENTS.md``.
Keep this file under 300 lines to preserve LLM context.


Skills
------

WarpX defines reusable *skills* in the ``.claude/skills/`` directory.
Skills are scripted workflows that an LLM assistant can execute on demand, automating multi-step tasks that follow a fixed procedure.

All WarpX skills use the prefix ``warpx-`` for easy discovery (start typing ``/warpx`` to see them).

Currently available skills:

``/warpx-answer-user-question``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Drafts a response to a WarpX user question from a GitHub issue, discussion, or email.

Usage (in Claude Code):

.. code-block:: text

   /warpx-answer-user-question https://github.com/BLAST-WarpX/warpx/discussions/1234

The skill will:

#. Fetch the question from the provided URL (or use a pasted question directly).
#. Categorize the question (installation, input parameters, diagnostics, physics, etc.).
#. Search the WarpX source code, documentation, and past issues for relevant information.
#. Draft a response in the style of an experienced WarpX developer.
#. Present the draft for review before posting.

``/warpx-new-paper-highlight``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adds a new paper to the WarpX science highlights documentation (``Docs/source/highlights.rst``).

Usage (in Claude Code):

.. code-block:: text

   /warpx-new-paper-highlight https://doi.org/10.1103/PhysRevLett.133.045002

The skill will:

#. Fetch the paper metadata (authors, title, journal, DOI) from the provided URL.
#. Choose the appropriate highlights section (e.g., Plasma-Based Acceleration, HPC and Numerics).
#. Format the entry in the RST style used in the file.
#. Create a branch, commit the change, and optionally open a pull request.

``/warpx-self-review-pr``
^^^^^^^^^^^^^^^^^^^^^^^^^

Reviews the changes on your current branch relative to ``development``, as a WarpX maintainer would review a pull request (see the :ref:`Self-Reviewing a Pull Request <developers-llm-self-review>` section below).

Usage (in Claude Code):

.. code-block:: text

   /warpx-self-review-pr

The skill will:

#. Read ``AGENTS.md`` for the project conventions.
#. Fetch the latest upstream ``development`` and diff your branch against it.
#. Review the diff against a checklist (correctness, algorithmic scaling, dimensionality, GPU/CPU portability, AMReX usage, backward compatibility, testing, style, auto-generated files, documentation, scope).
#. Report concrete findings with file and line references, each with a severity and a suggested fix.

To add new skills, create a directory under ``.claude/skills/<skill-name>/`` containing a ``SKILL.md`` file that describes the step-by-step procedure.


.. _developers-llm-self-review:

Self-Reviewing a Pull Request
-----------------------------

As emphasized above, you should carefully review all LLM-generated code *yourself* before requesting a review from other WarpX developers.
A coding assistant can help with this first pass: it can re-read your own diff with fresh context and flag issues you may have missed.
This does not replace your own critical review, but it makes that review more effective.

Before using the prompt below, commit your work and make sure your branch is up to date with ``development``, so that the diff the assistant reviews matches what reviewers will see.
Run it in a *fresh* session (not the one that wrote the code). If this session also wrote the code under review, say so up front and weigh your own prior choices skeptically.

The review is a static reading of the diff: the assistant is asked not to compile the code or run the test suite, since the CI checks do that once the pull request is open.
This keeps the pass fast and avoids spending a long local build on something CI reports anyway.
The prompt says so explicitly because ``AGENTS.md`` documents how to build and test WarpX for development work, and an assistant reading it for the project conventions would otherwise be inclined to do both.

The prompt also asks the assistant to check AMReX behavior against real source, at the ``commit_amrex`` pinned in ``dependencies.json``, rather than recalling APIs from memory, which is a common source of confident but wrong review findings.
Reading the pin costs nothing: a clone next to your WarpX checkout can show any file at that commit without touching your working tree or branches, and ``raw.githubusercontent.com`` serves it without a clone at all.
Local copies are not a substitute, since they drift: a sibling clone tracks AMReX ``development``, and the copy CMake fetched into ``build/_deps/fetchedamrex-src`` is frozen at whatever ``WarpX_amrex_branch`` was cached as, so neither follows a pin update.

What the prompt does *not* ask for is a verdict on whether the code compiles against the pin: CI checks out that commit and builds it with ``-Werror``, which settles signature mismatches and deprecated APIs far more reliably than reading headers.
The AMReX item on the checklist is therefore limited to what a compiler cannot judge, namely whether the code uses the right AMReX abstraction.

In Claude Code, the ``/warpx-self-review-pr`` skill runs this review for you.
For other assistants, adapt the prompt in the
`Claude Code skill <https://github.com/BLAST-WarpX/warpx/blob/development/.claude/skills/warpx-self-review-pr/SKILL.md>`__.


Documentation Context via MCP Servers
--------------------------------------

LLM assistants work best when they can query up-to-date project documentation.
The :ref:`AI-Assisted Input File Design <ai_input_design>` workflow page describes how to set up `Model Context Protocol (MCP) <https://modelcontextprotocol.io>`__ servers for this purpose.
That setup is equally useful for development tasks: the same documentation context that helps write input files also helps an assistant understand WarpX internals, AMReX and pyAMReX APIs, and conventions when writing C++ or Python code.

See :ref:`ai_input_design` for general MCP setup instructions with Context7.

WarpX and Dependency Documentation on Context7
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since WarpX builds on top of `AMReX <https://amrex-codes.github.io/amrex/>`__, `pyAMReX <https://pyamrex.readthedocs.io>`__, and `openPMD-api <https://openpmd-api.readthedocs.io>`__, providing documentation for these dependencies alongside WarpX documentation gives the assistant much richer context for development tasks.

The following documentation is available through Context7:

- **WarpX**: `context7.com/blast-warpx/warpx <https://context7.com/blast-warpx/warpx>`__
- **AMReX**: `context7.com/amrex-codes/amrex <https://context7.com/amrex-codes/amrex>`__
- **pyAMReX**: `context7.com/amrex-codes/pyamrex <https://context7.com/amrex-codes/pyamrex>`__
- **openPMD-api**: `context7.com/openpmd/openpmd-api <https://context7.com/openpmd/openpmd-api>`__
- **openPMD-viewer**: `context7.com/openpmd/openpmd-viewer <https://context7.com/openpmd/openpmd-viewer>`__
- **PICSAR-QED**: `context7.com/ecp-warpx/picsar <https://context7.com/ecp-warpx/picsar>`__
- **PICMI**: `context7.com/picmi-standard/picmi <https://context7.com/picmi-standard/picmi>`__
- **pybind11**: `context7.com/pybind/pybind11 <https://context7.com/pybind/pybind11>`__

When Context7 connected, the assistant can look up any of those when needed:
AMReX data structures (e.g., ``MultiFab``, ``ParticleContainer``, ``Geometry``), pyAMReX and pybind11 binding patterns, and openPMD I/O APIs directly, which is especially helpful when working, for instance, on:

- Field solver implementations that use AMReX mesh data structures
- Particle routines built on ``amrex::ParticleContainer``
- Python bindings that wrap C++ classes via pybind11 and pyAMReX
- I/O and diagnostic code that interacts with AMReX plotfiles or openPMD

For instructions on configuring Context7 as an MCP server in your coding assistant (Claude Code, Cursor, VS Code, Windsurf, Codex CLI, and others), see the `Context7 client documentation <https://context7.com/docs/resources/all-clients>`__ and the :ref:`ai_input_design` page.
