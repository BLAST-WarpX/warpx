---
name: warpx-new-paper-highlight
description: Add a new paper to the WarpX science highlights documentation (`Docs/source/highlights.rst`).
disable-model-invocation: true
---

# New Paper Highlight

Add a new paper to the WarpX science highlights documentation (`Docs/source/highlights.rst`).

## Step 1 — Get the paper URL

If the user has provided a paper URL (e.g. as `$ARGUMENTS`), use it.
Otherwise, ask the user:

> Please provide a URL or DOI link for the paper you want to add.

## Step 2 — Extract paper metadata

Fetch the URL with the WebFetch tool and extract:

- **Authors**: last name + initials, comma-separated (e.g. `Smith J, Doe A B`)
- **Title**: full paper title
- **Journal**: abbreviated journal name (match abbreviation style used in the file, e.g. `Phys. Rev. Lett.`, `Physics of Plasmas`, `The Astrophysical Journal`)
- **Volume**: journal volume number (bold it in RST with `**vol**`)
- **Issue / Article number / Pages**: whatever is available
- **Year**: publication year
- **DOI**: the DOI string (e.g. `10.1103/PhysRevLett.133.045002`)

If the page returns a 403 or is otherwise inaccessible, try fetching via `https://doi.org/<DOI>` if you can extract the DOI from the URL. If still unavailable, ask the user to paste the relevant bibliographic information.

## Step 3 — Choose the highlights section

Read `Docs/source/highlights.rst` to remind yourself of the available sections and their descriptions:

| Section | Description |
|---|---|
| Plasma-Based Acceleration | Laser-plasma and beam-plasma acceleration |
| Laser-Plasma Interaction | Laser-ion acceleration and laser-matter interaction |
| Particle Accelerator & Beam Physics | Particle and beam modeling |
| Astrophysical Plasma Physics | Astrophysical plasma modeling |
| Microelectronics | ARTEMIS / microelectronics |
| High-Performance Computing and Numerics | HPC, applied mathematics, numerics |
| Nuclear Fusion and Plasma Confinement | Fusion and plasma confinement |
| Plasma Thrusters and Spacecraft Physics | Hall thrusters, spacecraft |

Pick the most appropriate section based on the paper's abstract, keywords, and journal. If a paper could fit two sections, choose the most specific one. Show the user your choice and reasoning in one sentence.

## Step 4 — Format the RST entry

Format the new entry following the style already used in the file:

```rst
#. LastName1 F, LastName2 A B, LastName3 C.
   **Full paper title here**.
   Journal Name **volume**, article-or-page, year.
   `DOI:10.xxxx/xxxxx <https://doi.org/10.xxxx/xxxxx>`__
```

Formatting rules:
- Author list: `LastName Initials` separated by commas, last author uses `and` before the name (e.g. `Smith J, Doe A and Jones B`). No `and` needed if there is only one author.
- Title: sentence case (capitalise only the first word and proper nouns), wrapped in `**...**`.
- Journal: use standard abbreviations consistent with the rest of the file.
- Volume in bold: `**27**`.
- Omit fields that are not available (e.g. no issue number if not present).
- If a preprint link is also available, add it as a separate inline link before the DOI link:
  `` `preprint <https://arxiv.org/abs/...>`__, `` followed by the DOI link.
- New entries go at the **top** of the chosen section (after the section heading and its description line), so they appear in reverse-chronological order.

## Step 5 — Create a git branch and edit the file

1. Make sure the local `development` branch is up to date:
   ```bash
   git fetch origin development
   ```

2. Create and check out a new branch named `new_paper` from `development`:
   ```bash
   git checkout -b new_paper origin/development
   ```
   If a branch called `new_paper` already exists, append a short slug from the paper title or DOI, e.g. `new_paper_smith2025`.

3. Edit `Docs/source/highlights.rst` using the Edit tool to insert the formatted entry at the top of the chosen section.

4. Show the user a diff of the change (use `git diff Docs/source/highlights.rst`).

## Step 6 — Offer to open a PR

Ask the user:

> Would you like me to push this branch and open a pull request against `development`?

If the user agrees:
1. Commit the change and push the user's personal fork:
2. Open the PR with `gh`:
   ```bash
   source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && \
   gh pr create \
     --base development \
     --title "Doc: new paper in highlights – <Short title or first author + year>" \
     --body "$(cat <<'EOF'
   ## Summary

   - Adds **<Paper title>** by <First author> et al. (<year>) to the *<Section name>* section of the science highlights.
   - DOI: https://doi.org/<DOI>

   🤖 Generated with [Claude Code](https://claude.com/claude-code)
   EOF
   )"
   ```
3. Return the PR URL to the user.
