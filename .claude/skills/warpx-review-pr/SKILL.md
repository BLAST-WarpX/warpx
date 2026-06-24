---
name: warpx-review-pr
description: Review a WarpX pull request in the style of an experienced WarpX maintainer.
disable-model-invocation: true
---

# Review a WarpX Pull Request

Review a WarpX pull request the way experienced WarpX developers actually review code.

Aim for the kind of feedback an experienced WarpX developer would leave: targeted, grounded in the repo, and clearly separated into "blocks merging" vs. "follow-up PR is fine."

## Step 1 — Get the PR

If a PR number/URL was passed as `$ARGUMENTS`, use it. Otherwise ask:

> Which PR should I review? (number or URL — e.g. `6789` or `https://github.com/BLAST-WarpX/warpx/pull/6789`)

Fetch the PR with `gh` (activate base if needed: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate base`):

```bash
gh pr view <N> --repo BLAST-WarpX/warpx --json number,title,body,author,baseRefName,headRefName,additions,deletions,changedFiles,labels,files
gh pr diff <N> --repo BLAST-WarpX/warpx
gh api repos/BLAST-WarpX/warpx/pulls/<N>/commits --jq '.[] | {sha: .sha[0:8], message: .commit.message}'
```

Also read existing review threads to avoid re-raising what others have already covered:

```bash
gh api repos/BLAST-WarpX/warpx/pulls/<N>/comments --jq '.[] | {user: .user.login, path: .path, line: .line, body: .body}'
gh api repos/BLAST-WarpX/warpx/issues/<N>/comments --jq '.[] | {user: .user.login, body: .body}'
```

## Step 2 — Understand what the PR does

Before opening any review file, answer in your own head:

- **What is the user-facing change?** New input parameter? Renamed/removed one? New algorithm/diagnostic/solver?
- **Which subsystems does it touch?** (laser, particle pusher, field solver, EB, RZ, PSATD, hybrid-PIC, …)
- **Which dimensionalities/geometries?** Cartesian vs RZ vs RCYLINDER/RSPHERE; single vs multi-mode RZ.
- **Which backends?** CPU/OMP, CUDA, HIP, SYCL — does the change touch GPU kernels?
- **Is this a bug fix, refactor, new feature, or breaking change?**

If the PR description doesn't make this clear, that itself is feedback (ask the author to expand the description).

## Step 3 — Read the diff strategically

Read the diff with these priorities, in order:

1. **Public surface**: header files, input-parameter parsing (`ParmParse`, `query`, `queryWithParser`, `queryAdd`), PICMI bindings, `Source/Python/`, public docs.
2. **Algorithmic core**: any new physics/numerics — read enough to understand the structure.
3. **Tests added/changed**: confirm they actually exercise the new code path (not just compile it).
4. **Auto-generated files**: skip `.pyi` stubs, `dependencies.json`, and `Regression/Checksum/benchmarks_json/*.json` — these are auto-updated and not human-reviewed.

## Step 4 — Walk the six review focus areas

Evaluate each area; collect findings as you go. Use the **gating** rule: a PR cannot merge if **A** or **B** are unmet — flag those as blockers. **C**–**F** are typically change-requests, suggestions, or follow-up PRs.

### A. Documentation (gating)

Run these checks:

- If the PR adds/renames a **user-facing input parameter**, it must appear in `Docs/source/usage/parameters.rst`. Grep for the parameter name in `Docs/source/` to verify.
- If the PR exposes a new **Python/PICMI** API, check `Docs/source/usage/python.rst`.

For Sphinx cross-references, prefer `:pp:param:`/`:ref:` over bare paragraph text.

### B. CI testing (gating)

Run these checks:

- Does the PR add or modify a test under `Examples/Tests/` (or a `CMakeLists.txt` in that tree)? New features need a test added via `add_warpx_test()`.
- Does the new test **actually exercise the new code path**? A red flag is when the new code is reachable but no checksum file for any test changed — that suggests no test enters the new branch.
- If the change is dimensionality- or backend-specific (e.g., works only in RZ, only on CUDA), is there a test for that configuration? Especially for **MR + new feature** combinations.
- If CI is red on the PR, do not dismiss it. Ask the author to investigate (or post a local-vs-CI tolerance comparison) rather than silently bumping tolerances.
- Tests must be quick (<~30s on a 2-core CI runner).

**Do not approve tolerance increases without evidence.** If a tolerance is being raised, the author should provide a plot or a before/after comparison showing the relaxation is physically justified.

### C. Consistency with the rest of WarpX

This is the area where reviewer experience matters most. Look for:

- **Input-parameter naming.** New parameter names should match WarpX conventions: lowercase with dots/underscores, grouped by prefix (`warpx.`, `algo.`, `particles.`, `<species>.`, `<diag_name>.`, …). Names should reflect semantics (`@roelof-groenewald` PR #6756 renamed `num_redistribute_ghost` → `max_num_cells_travelled`). Grep `Docs/source/usage/parameters.rst` for similar parameters to compare style.
- **Code duplication.** Before approving a new helper function, grep `Source/` for an existing one. Canonical example: `@dpgrote` on PR #6583 — *"There is already a wrapper for `DefaultInitializeRuntimeAttributes`. Can this be reused?"*
- **Symmetric handling.** If a parameter exists in one path (e.g. `AddPlasma`), check whether the analog (`AddPlasmaFlux`) has the same handling. Asymmetry is usually an oversight.
- **Breaking changes.** If a user-facing parameter is removed or renamed:
  - The PR **must** add a guard to `WarpX::BackwardCompatibility()` in `Source/WarpX.cpp` (or the equivalent method in `PhysicalParticleContainer.cpp`) emitting a clear error message pointing at the new syntax.

### D. Readability and long-term maintainability

Ask: *would another experienced WarpX developer be able to understand this code in a year?* Look for:

- **In-code comments on non-obvious choices** — e.g., why a particular volume scaling, why a specific stencil width, why a particular AMReX `nGrowVect`. Code that encodes physics/numerics conventions should say so.
- **References to papers/equations** for any new numerical scheme — citation in a comment or in `Docs/source/theory/`.
- **Asserts on assumptions.** This is one of the most common review asks. If the new code only works in a specific configuration, add `WARPX_ABORT_WITH_MESSAGE(...)` or `WARPX_ALWAYS_ASSERT_WITH_MESSAGE(cond, "...")` rather than letting users hit silent wrong answers. Common assumption asserts:
  - "Cartesian only" / "RZ only" (`WarpX::n_rz_azimuthal_modes == 1` for single-mode-only code)
  - "Single level only" / "no mesh refinement"
  - "Photon mass must be zero" / "non-zero mass required" (PR #6265)
  - "Implicit solver only" / "explicit solver only"
  - Watch out for **over-broad asserts**: an assert added in a shared path may incorrectly gate code that works for cases the author didn't think about. The error message must match the actual gated case.
- **`amrex::` namespace prefix.** New code should not rely on `using namespace amrex;` — prefix types as `amrex::Real`, `amrex::MultiFab`, etc.
- **No magic numbers / unstated assumptions.** Hard-coded particle-shape orders, hard-coded centerings, hard-coded grow factors — call them out.

### E. Performance and GPU portability

You are not expected to profile the code, but flag obvious red flags:

- **Unneeded memory allocation** inside hot loops (per-iteration `MultiFab` creation, per-particle `std::vector` growth).
- **Added communication** — new `FillBoundary`, `ParallelCopy`, MPI reductions, or `Redistribute` calls. Question whether each one is necessary and whether it lives outside the time-step inner loop.
- **Kernel size growth.** GPU kernels that grow large (lots of conditional branches, large register pressure) hurt occupancy. If a kernel is becoming a switchboard over many physics options, suggest splitting via `constexpr if`, template dispatch, or function objects.
- **CPU fallbacks on GPU builds** are generally not OK. `@RemiLehe` (PR #168): *"I think that it's okay to prevent users from using routines that will run on the CPU. (We don't want them to use them by accident without knowing that they will be slow.)"* Prefer a hard abort over a silent slow path.

### F. Style basics (quick wins)

The pre-commit hook handles formatting, but reviewers still call out:

- 4 spaces, no tabs, ≤ 100 chars/line
- Space before `(` in function **definitions**, no space in **calls** (so `git grep "foo ()"` distinguishes them — see `Docs/source/developers/contributing.rst`)
- Opening brace `{` on a new line for functions/classes
- Curly braces around all single-statement blocks
- Include order: corresponding header → WarpX headers → WarpX forward decls → AMReX headers → AMReX forward decls → third-party → standard library, each group alphabetical with blank lines between

If formatting issues are everywhere, comment once at the top of the PR rather than line-by-line — pre-commit will catch them.

## Step 5 — Compose the review

Group findings by severity. Use this template:

```
## Review of #<N>: <title>

### Summary
<1–3 sentences: what the PR does and your overall take. Be direct: "This looks good but needs X before merging," or "Looks good, minor suggestions below.">

### Blockers (must fix before merging)
- [ ] **Documentation**: <param/feature> needs an entry in `Docs/source/usage/parameters.rst`.
- [ ] **CI test**: please add a test that exercises <feature> under <configuration>.
<…>

### Change requests
- **<topic>** (`<file>:<line>`): <ask, with 1-sentence rationale>
<…>

### Suggestions / nits (non-blocking)
- <minor naming/clarity point>
<…>

### Follow-up PR (track separately)
- Add `WARPX_ABORT_WITH_MESSAGE` for <unsupported config> in `<file>:<line>` — non-blocking but should land before the next release.
<…>

### Questions
- <clarification question for the author>
```

**Tone**: polite, direct, concrete.
- *"This looks good, but…"*
- *"Could this be reused from `<file>`?"*
- *"Something for a future PR…"*
- *"Could you double-check why the automated tests are failing?"*

Always reference **specific files and line numbers**. Always explain **why**, not just "change this." Distinguish must-fix from nice-to-have so the author can prioritize.

## Step 6 — Present the draft

Show the draft to the user (the WarpX developer running this skill) and ask:

> Does this review look good? Would you like me to post it to the PR, or would you like to adjust anything first?

**Never post the review to GitHub without explicit confirmation.** When confirmed, post with:

```bash
gh pr review <N> --repo BLAST-WarpX/warpx --comment --body "$(cat <<'EOF'
<review body>
EOF
)"
```

(Use `--request-changes` only if the user explicitly asks for that — the WarpX default register is comment-style, with blockers called out in prose.)
