#!/usr/bin/env bash

set -o nounset
set -o errexit
set -o pipefail

# Parse command line arguments
head_ref=${1}
base_ref=${2}
clone_url=${3}

# Set paths to ignore
paths_ignore="^(Docs|\.github)/|\.azure-pipelines\.yml$"

# Add forked repository as remote
git remote add fork ${clone_url}

# Fetch base branch from main repository
git fetch origin ${base_ref}

# Fetch head branch from forked repository
git fetch fork ${head_ref}

# Save output of git diff to inspect files changed
git diff --name-only --diff-filter=ACMRTUXB origin/${base_ref}..fork/${head_ref} > check_diff.txt

# Set skip variable after inspecting files changed
skip=$(grep -v -E "${paths_ignore}" check_diff.txt)

# Set an environment variable based on the output
if [ -z "$skip" ]; then
  echo "SKIP_CHECKS=true" >> $GITHUB_ENV
else
  echo "SKIP_CHECKS=false" >> $GITHUB_ENV
fi
