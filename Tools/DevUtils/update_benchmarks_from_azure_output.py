# Copyright 2023 Neil Zaim, Edoardo Zoni
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
This Python script updates the Azure benchmarks automatically.

It can be run in two ways:

1. Automatic mode (recommended): pass a GitHub PR number, and the script
   fetches the failing Azure Pipelines logs automatically.

       python update_benchmarks_from_azure_output.py --pr-number 1234

   The ``gh`` CLI must be installed and authenticated (available via conda base).

2. Legacy mode: pass a raw Azure log text file downloaded manually.

       python update_benchmarks_from_azure_output.py path/to/raw_log.txt
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request

# path of all checksums benchmark files, relative to this script
benchmark_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../Regression/Checksum/benchmarks_json/",
)

# string to identify failing tests that require a checksums reset
new_checksums_marker = "New checksums"


def _detect_prefix_length(first_line):
    """Return the length of the Azure timestamp prefix on a log line.

    Azure raw logs have lines like:
        2024-01-01T00:00:00.0000000Z ##[section]Starting: ...
    The prefix is everything before the first '#'.  If no '#' is found,
    assume there is no prefix (length 0).
    """
    idx = first_line.find("#")
    return idx if idx >= 0 else 0


def update_benchmarks_from_log_text(log_text):
    """Parse Azure log text and update checksum JSON files for failing tests.

    Returns a list of JSON filenames that were updated.
    """
    lines = log_text.splitlines()
    if not lines:
        return []

    prefix_length = _detect_prefix_length(lines[0])

    failing_test = ""
    json_file_string = ""
    updated = []

    for line in lines:
        # remove Azure timestamp prefix
        line = line[prefix_length:]

        if failing_test == "":
            # no failing test found yet
            if re.search(new_checksums_marker, line):
                # found a failing test – extract its name
                failing_test = line[line.find("test_") : line.find(".json")]
                json_file_string = ""
        else:
            # accumulate JSON lines for the failing test
            json_file_string += line + "\n"
            if line.startswith("}"):  # end of new checksums block
                json_file = json.loads(json_file_string)
                json_filename = failing_test + ".json"
                json_filepath = os.path.join(benchmark_path, json_filename)
                print(f"\nDumping new checksums file {json_filename}:")
                print(json_file_string)
                with open(json_filepath, "w") as json_f:
                    json.dump(json_file, json_f, indent=2)
                updated.append(json_filename)
                # reset to continue searching for more failing tests
                failing_test = ""

    return updated


# ---------------------------------------------------------------------------
# Automatic mode: fetch logs from Azure via GitHub PR number
# ---------------------------------------------------------------------------


def get_azure_build_info(pr_number, repo):
    """Use ``gh`` to find the Azure Pipelines build URL for a PR.

    Returns (org, project, build_id).
    """
    print(f"Fetching CI checks for PR #{pr_number} on {repo} ...")
    result = subprocess.run(
        [
            "gh",
            "pr",
            "checks",
            str(pr_number),
            "--repo",
            repo,
            "--json",
            "name,state,link",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error running 'gh pr checks': {result.stderr}", file=sys.stderr)
        sys.exit(1)

    checks = json.loads(result.stdout)

    # Find the top-level BLAST-WarpX.CI check that links to the Azure build
    azure_url = None
    for check in checks:
        details_url = check.get("link", "")
        if check["name"] == "BLAST-WarpX.CI" and "dev.azure.com" in details_url:
            azure_url = details_url
            break

    if azure_url is None:
        print(
            "Could not find an Azure Pipelines build URL in the PR checks.\n"
            "Make sure the CI has run and that 'BLAST-WarpX.CI' appears in the checks.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse https://dev.azure.com/{org}/{project}/_build/results?buildId={id}
    m = re.match(
        r"https://dev\.azure\.com/([^/]+)/([^/]+)/_build/results\?buildId=(\d+)",
        azure_url,
    )
    if not m:
        print(f"Could not parse Azure build URL: {azure_url}", file=sys.stderr)
        sys.exit(1)

    org, project, build_id = m.group(1), m.group(2), m.group(3)
    print(f"Found Azure build: org={org}, buildId={build_id}")
    return org, project, build_id


def get_failing_test_log_ids(org, project, build_id):
    """Query the Azure DevOps build timeline and return log IDs for failing Test tasks.

    Returns a list of (log_id, job_name) tuples.
    """
    url = (
        f"https://dev.azure.com/{org}/{project}"
        f"/_apis/build/builds/{build_id}/timeline?api-version=7.1"
    )
    print("Fetching build timeline from Azure ...")
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req) as resp:
        timeline = json.loads(resp.read().decode())

    records = timeline.get("records", [])

    # Build a map from record id → name for parent look-ups
    id_to_name = {r["id"]: r.get("name", "unknown") for r in records}

    failing = []
    for record in records:
        if (
            record.get("type") == "Task"
            and record.get("name") == "Test"
            and record.get("result") == "failed"
            and record.get("log")
        ):
            log_id = record["log"]["id"]
            parent_name = id_to_name.get(record.get("parentId", ""), "unknown")
            print(f"  Failing Test task in job '{parent_name}', log ID: {log_id}")
            failing.append((log_id, parent_name))

    if not failing:
        print("No failing Test tasks found in the Azure build timeline.")

    return failing


def download_azure_log(org, project, build_id, log_id):
    """Download a raw log from Azure DevOps and return it as a string."""
    url = (
        f"https://dev.azure.com/{org}/{project}"
        f"/_apis/build/builds/{build_id}/logs/{log_id}?api-version=7.1"
    )
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8", errors="replace")


def process_pr(pr_number, repo):
    """Fetch Azure logs for a PR and update all failing checksum benchmarks."""
    org, project, build_id = get_azure_build_info(pr_number, repo)
    failing_log_ids = get_failing_test_log_ids(org, project, build_id)

    if not failing_log_ids:
        print("No failing tests to update.")
        return

    all_updated = []
    for log_id, job_name in failing_log_ids:
        print(f"\nDownloading log for job '{job_name}' (log ID: {log_id}) ...")
        log_text = download_azure_log(org, project, build_id, log_id)
        updated = update_benchmarks_from_log_text(log_text)
        all_updated.extend(updated)

    if all_updated:
        print(f"\nSuccessfully updated {len(all_updated)} checksum file(s):")
        for f in all_updated:
            print(f"  {f}")
    else:
        print("\nNo checksum files needed updating.")


# ---------------------------------------------------------------------------
# Legacy mode: process a downloaded Azure log file
# ---------------------------------------------------------------------------


def process_file(azure_output):
    """Parse a manually downloaded Azure log file and update checksum benchmarks."""
    with open(azure_output, "r") as f:
        log_text = f.read()
    update_benchmarks_from_log_text(log_text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Update WarpX checksum benchmarks from Azure Pipelines output. "
            "Either provide a GitHub PR number (automatic mode) or a path to "
            "a raw Azure log file (legacy mode)."
        )
    )
    parser.add_argument(
        "--pr-number",
        type=int,
        metavar="PR",
        help=(
            "GitHub PR number. The script will automatically fetch the failing "
            "Azure Pipelines logs and update the checksum benchmark files. "
            "Requires the 'gh' CLI to be installed and authenticated."
        ),
    )
    parser.add_argument(
        "--repo",
        default="BLAST-WarpX/warpx",
        metavar="OWNER/REPO",
        help="GitHub repository (default: BLAST-WarpX/warpx).",
    )
    parser.add_argument(
        "azure_output",
        nargs="?",
        metavar="LOG_FILE",
        help="Path to a raw Azure log text file (legacy mode).",
    )

    args = parser.parse_args()

    if args.pr_number is not None and args.azure_output is not None:
        parser.error("Provide either --pr-number or a log file path, not both.")

    if args.pr_number is not None:
        process_pr(args.pr_number, args.repo)
    elif args.azure_output is not None:
        process_file(args.azure_output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
