#!/usr/bin/env bash

set -o nounset
set -o errexit
set -o pipefail
set -o xtrace

# Parse command line arguments
head_ref=${1}
base_ref=${2}
clone_url=${3}

# Set paths to ignore
paths_ignore="^(Docs|\.github)/|\.azure-pipelines\.yml$"

# Check if remote repository already exists
if ! git remote --get-url fork > /dev/null 2>&1; then
  # Add forked repository as remote
  git remote add fork ${clone_url}
fi

# Fetch base branch from main repository
git fetch origin ${base_ref}

# Fetch head branch from forked repository
git fetch fork ${head_ref}

# Save output of git diff to inspect files changed
git diff --name-only --diff-filter=ACMRTUXB origin/${base_ref}..fork/${head_ref} > check_diff.txt

# Check if check_diff.txt is not empty
if [ -s check_diff.txt ]; then
  # Set skip variable after inspecting files changed
  skip=$(grep -v -E "${paths_ignore}" check_diff.txt)
  # Set an environment variable based on the output
  if [ -z "$skip" ]; then
    echo "SKIP_CHECKS=true" >> $GITHUB_ENV
  else
    echo "SKIP_CHECKS=false" >> $GITHUB_ENV
  fi
else
  # If check_diff.txt is empty, set SKIP_CHECKS to true
  echo "SKIP_CHECKS=true" >> $GITHUB_ENV
fi
