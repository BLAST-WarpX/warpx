.. _ai_input_design:

AI (LLM)-Assisted Input File Design
===================================

Large Language Models (LLMs) can accelerate the process of creating and modifying WarpX input files and Python (PICMI) scripts.
By providing an LLM-based coding assistant with WarpX documentation as context, users can describe a simulation setup in natural language and receive a draft parameter list file or Python script, ask for explanations of existing parameters, or request modifications to an existing configuration.

This workflow is equally applicable to:

- **WarpX parameter list files** (:ref:`parameter list files <running-cpp-parameters>`)
- **Python/PICMI scripts** (:ref:`Python scripts <usage-picmi>`)

.. note::

   LLM-generated input files and scripts should always be reviewed carefully before use.
   LLMs tend to hallucinate options that do not exist or might miss the mental context of your physics.
   The guide below tries to improve this situation by providing LLMs all WarpX examples and documentation automatically, but this remains an active and rapidly evolving field of tooling.

   Validate physics parameters, boundary conditions, grid resolution, and numerical settings against your domain knowledge and the `WarpX documentation <https://warpx.readthedocs.io>`__.


.. note::

   This section is not understood as an endorsement of any of any of the listed (or unlisted) coding assistants or MCP services.
   Contributions to this section documenting further services, clients, skills, etc are encouraged.

How It Works: MCP Servers
-------------------------

`Model Context Protocol (MCP) <https://modelcontextprotocol.io>`__ servers are a standardized way to provide external context, such as library documentation, to LLM-based coding assistants.
When an MCP server is configured, the assistant can query up-to-date WarpX documentation on demand, rather than relying solely on its training data.

Setting Up Context7 as an MCP Server
------------------------------------

`Context7 <https://context7.com>`__ is a service that indexes open-source project documentation and serves it through the MCP protocol.
WarpX documentation is available at:

    `context7.com/blast-warpx/warpx <https://context7.com/blast-warpx/warpx>`__

When writing Python/PICMI scripts, users will also encounter `pyAMReX <https://pyamrex.readthedocs.io>`__ APIs (e.g., for accessing mesh and particle data, callbacks, or custom field initialization):

- **pyAMReX**: `context7.com/amrex-codes/pyamrex <https://context7.com/amrex-codes/pyamrex>`__

To configure Context7 for your coding assistant, pick your client below or see the `Context7 documentation <https://context7.com/docs/resources/all-clients>`__ for further clients.

.. dropdown:: Client configuration examples
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. tab-set::

      .. tab-item:: Claude Code

         Add the Context7 MCP server from the command line:

         .. code-block:: bash

            claude mcp add context7 https://mcp.context7.com/mcp \
                --scope user --transport http \
                --header "CONTEXT7_API_KEY: YOUR_API_KEY"

      .. tab-item:: Codex

         Add to ``~/.codex/config.toml``:

         .. code-block:: toml

            [mcp_servers.context7]
            url = "https://mcp.context7.com/mcp"
            http_headers = { "CONTEXT7_API_KEY" = "YOUR_API_KEY" }

      .. tab-item:: Cursor

         Add to ``~/.cursor/mcp.json`` (all projects) or ``.cursor/mcp.json`` (per project):

         .. code-block:: json

            {
              "mcpServers": {
                "context7": {
                  "url": "https://mcp.context7.com/mcp",
                  "headers": {
                    "CONTEXT7_API_KEY": "YOUR_API_KEY"
                  }
                }
              }
            }

      .. tab-item:: More Clients

         VS Code Copilot, Windsurf, JetBrains IDEs, Gemini CLI, and many other clients are covered in the `Context7 client documentation <https://context7.com/docs/resources/all-clients>`__.

Once connected, a coding assistant (Claude Code, Cursor, VS Code Copilot, Windsurf, etc.) can retrieve relevant sections of all the above projects in real time when helping you write or debug input files and PICMI scripts.

.. tip::

   Some clients support optional API key authentication.
   See the `Context7 client docs <https://context7.com/docs/resources/all-clients>`__ for details on API keys and OAuth options.

After creating a personal account, Context7 is free within limits for open source projects.

.. tip::

   **Berkeley Lab (LBNL) users:** Context7 is available under a commercial license through the laboratory's `CBorg AI Portal <https://cborg.lbl.gov/mcp_servers/>`__.
   CBorg provides Context7 via its MCP gateway, which is compatible with any MCP-aware client (e.g., Claude Code, Cursor).
   In the configurations above, replace the server URL with ``https://api.cborg.lbl.gov/mcp/`` and the API-key header with ``Authorization: Bearer YOUR_CBORG_API_KEY`` (`request a CBorg API key <https://cborg.lbl.gov/api_request/>`__).
   Access requires LBLnet/VPN or registering your IP address by logging in at `api.cborg.lbl.gov/key/manage <https://api.cborg.lbl.gov/key/manage>`__.
   Under the commercial license, the provider will not use your queries for training.

Best Practices
--------------

When working with LLM coding assistants, keep in mind that *"most best practices are based on one constraint: [the] context window fills up fast, and performance degrades as it fills"* (`Claude Code Best Practices <https://code.claude.com/docs/en/best-practices>`__).
Starting from examples and iterating incrementally as described below and making a plan with the LLM assistant helps keep sessions focused and productive.

#. **Start from examples.**
   Run your coding agent inside the WarpX source directory.
   Point the assistant to an existing input file or PICMI script from the ``Examples/`` directory and ask it to adapt the setup to your needs.

#. **Iterate incrementally.**
   Start with a minimal working configuration, verify it runs, then ask the assistant to add complexity (diagnostics, additional species, field solvers, etc.).

#. **Cross-check with documentation.**
   Use the assistant to look up parameter meanings and valid ranges in the WarpX docs, especially for numerical parameters like solvers, resolution, CFL, and filter settings.

#. **Validate before production runs.**
   Always run a short test simulation and inspect the output before committing to a large-scale run.
   The ``Examples/`` directory contains analysis scripts that can serve as templates for validation.
