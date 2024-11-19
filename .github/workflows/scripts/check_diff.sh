#!/usr/bin/env bash

set -o nounset
set -o errexit
set -o pipefail

# Parse command line arguments
head_ref=${1}
base_ref=${2}
clone_url=${3}

# Add forked repository as remote
git remote add fork ${clone_url}

# Fetch base branch from main repository
git fetch origin ${base_ref}

# Fetch head branch from forked repository
git fetch fork ${head_ref}

# Save output of git diff to inspect files changed
git diff --name-only --diff-filter=ACMRTUXB origin/${base_ref}..fork/${head_ref} > check_diff.txt
echo "Files changed:"
cat check_diff.txt

# Set paths to ignore
paths_ignore=()
paths_ignore+=("Docs/")
paths_ignore+=(".github/")
paths_ignore+=(".azure-pipelines.yml")
echo "Paths to ignore:"
echo ${paths_ignore}

# Set string for grep command
paths_ignore_string=$(IFS='|'; echo "${paths_ignore[*]}")

# Set skip variable after inspecting files changed
if ! grep -qEv "^(${paths_ignore_string})" check_diff.txt; then
  echo "SKIP_CHECKS=true" >> ${GITHUB_ENV}
else
  echo "SKIP_CHECKS=false" >> ${GITHUB_ENV}
fi
echo "GITHUB_ENV = ${GITHUB_ENV}"
