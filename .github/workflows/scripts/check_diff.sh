#!/usr/bin/env bash

set -o nounset
set -o errexit
set -o pipefail

# Parse command line arguments
head_ref=${1}
base_ref=${2}
clone_url=${3}

# Add forked repository as remote
# Fetch base branch from main repository
# Fetch head branch from forked repository
git remote add fork ${clone_url}
git fetch origin ${base_ref}
git fetch fork ${head_ref}

# Save output of git diff to inspect files changed
git diff --name-only --diff-filter=ACMRTUXB origin/${base_ref}..fork/${head_ref} > check_diff.txt
echo "Files changed:"
cat check_diff.txt

# Set paths to ignore (please test grep command below when adding new patterns)
paths_ignore=()
paths_ignore+=("Docs/")
paths_ignore+=(".github/")
echo "Paths to ignore:"
echo ${paths_ignore[@]}

# Set string for regular expression to grep
paths_ignore_string=$(IFS='|'; echo "${paths_ignore[*]}")

# Set skip variable after inspecting files changed
if ! grep -qEv "^(${paths_ignore_string})" check_diff.txt; then
  echo "Skip checks"
  echo "SKIP_CHECKS=true" >> ${GITHUB_ENV}
else
  echo "Run checks"
  echo "SKIP_CHECKS=false" >> ${GITHUB_ENV}
fi
