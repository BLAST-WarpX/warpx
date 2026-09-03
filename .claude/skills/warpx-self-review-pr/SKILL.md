---
name: warpx-self-review-pr
description: Review the changes on the current branch relative to `development`, as a WarpX maintainer would review a pull request.
disable-model-invocation: true
---

# Self-Review a Pull Request

Review the changes on the current branch relative to the `development` branch, as if you were a WarpX maintainer reviewing a pull request.
This is a first pass that helps the author catch issues before requesting a review from other WarpX developers.
It does not replace the author's own critical review.

For best results, this skill should be run in a *fresh* session (not the one that wrote the code). If this session also wrote the code under review, say so up front and weigh your own prior choices skeptically.
The author should commit their work and make sure their branch is up to date with `development`, so the diff the assistant reviews matches what reviewers will see.

The review is a static reading of the diff: do not build the code or run tests here, since compilation and the test suite are covered by the CI checks that run on the open pull request.
The procedure below says so explicitly because `AGENTS.md` documents how to build and test WarpX for development work, not for this review.

## Review procedure

Review the changes on my current branch relative to the `development`
branch, as if you were a WarpX maintainer reviewing my pull request.
Do not make any changes yet; report your findings first.

This is a read-only review of the source. Do not configure or compile
the code (no `cmake`, no `pip install`) and do not run the test suite
(no `ctest`, no analysis scripts): compilation, tests, and checksums are
covered by the CI checks that run once the pull request is open. Judge
the diff by reading it, not by executing it. The `Build Commands` and
`Testing` sections of AGENTS.md describe development work in general and
do not apply to this review: ignore them here.

Start by reading AGENTS.md for the project conventions (style,
portability, dimensionality, backward compatibility), then identify the
remote that points at github.com/BLAST-WarpX/warpx with `git remote -v`
(the WarpX contributing guide names it `mainline`; it may also be
`origin` or `upstream`). Call it <upstream>, then run
`git fetch <upstream> development` followed by
`git diff <upstream>/development...HEAD` to see my changes. Fetching
first ensures the diff is taken against the latest upstream
`development`, not a stale local copy.

To learn the intended purpose of the changes, read the branch's commit
messages (`git log <upstream>/development..HEAD`), and `gh pr view` if a
pull request is already open. Use the diff to locate the changes, then
read the full changed files around each hunk before judging them; three
lines of diff context is rarely enough to judge correctness.

When a finding depends on AMReX behavior, ground it in real source
rather than recalling it from memory, and read that source at the commit
pinned as `commit_amrex` in `dependencies.json`, which is the AMReX
version that CI builds against. Either way works, and the second needs
no clone:

Reading the pin costs nothing: a clone next to your WarpX checkout can
show any file at that commit without touching your working tree or
branches, and `raw.githubusercontent.com` serves it without a clone at
all.

- `git -C ../amrex show <pin>:Src/...`, if that clone exists and has the
  commit (run `git -C ../amrex fetch origin` if it does not). Keep it
  read-only: never check out, switch, or pull in that clone, since it
  may hold my own work.
- `https://raw.githubusercontent.com/AMReX-Codes/amrex/<pin>/Src/...`,
  which serves any file at that exact commit.

Do not judge whether the code compiles against the pin: CI does that
with `-Werror`. If you cannot reach AMReX source at the pin, fall back
to the AMReX documentation and mark the finding as unverified.

Check the following and report concrete issues with file and line references:

1. Correctness: logic errors, off-by-one/index mistakes, uninitialized
   values, incorrect physics or units, wrong sign conventions.
2. Algorithmic scaling: flag book-keeping or data-structure logic with
   worse asymptotic complexity than necessary, e.g. an O(N^2) loop over
   lists where an O(N) or O(N log N) approach exists.
3. Dimensionality: does the code handle all relevant builds
   (1D, 2D/XZ, 3D, RZ) correctly, including the compile-time macros?
4. GPU/CPU portability: any particle-to-grid deposition, scatter-add,
   histogram, or shared-counter loop must use `amrex::For` (not
   `amrex::ParallelFor`). Flag atomics that do not actually make a
   `ParallelFor` safe. See Docs/source/developers/portability.rst.
5. AMReX usage: is this the right AMReX abstraction, or does it
   hand-roll something AMReX already provides? Flag misuse that would
   still compile, e.g. wrong ghost-cell or index-type conventions.
6. Backward compatibility: if a user-facing input parameter was removed
   or renamed, is there a guard in the relevant BackwardCompatibility()?
7. Testing: is there a test covering the new feature? Judging
   from the input file and analysis script alone (without running it),
   does it look fast enough for a 2-core CI runner and written portably?
8. Style: does the diff follow the C++/Python style in AGENTS.md, and
   does it avoid reformatting unrelated code?
9. Auto-generated files: flag any manual edits to `.pyi` stubs,
   `dependencies.json`, or `Regression/Checksum/benchmarks_json/*.json`.
10. Documentation: are new user-facing parameters or features documented?
11. Scope: is anything unrelated to the stated purpose of the PR included?

For each finding, state the severity (blocking / should-fix / nit) and
suggest a concrete fix. End with a short summary of whether this PR looks
ready for human review.

## Presenting the findings

Treat the output as a checklist of things the author should verify themselves, not as a verdict: it may raise false positives or miss real problems.
Confirm each finding against the code before acting on it.
