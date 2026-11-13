#!/usr/bin/env python3
#
# Copyright 2026 The WarpX Community
#
# License: BSD-3-Clause-LBNL

"""Request reviews from the code owners of a pull request, or ping them.

GitHub requests reviews automatically only from code owners with write access
to the repository. This script handles all owners listed in CODEOWNERS: it
requests a review where GitHub allows it (members of the GitHub organization,
e.g., with triage access) and mentions the remaining owners in a comment that
explains why they are pinged.

The script only acts on pull requests that are ready for review (not a draft,
title does not start with `[WIP]`) and that nobody has organized reviews for
yet. It does nothing if a review is requested from a person or team by someone
other than GitHub (for CODEOWNERS) or this script, if a person other than the
author has reviewed (bots such as CodeQL or Copilot do not count), or if this
script commented before. So it comments at most once per pull request, since
every further comment would notify everyone pinged before. If a run fails half
way, a later run completes it.

Owners are contacted at most once per pull request, as one would do by hand.
Owners that the pull request's timeline shows as requested before (also if the
request was removed later, e.g., by the owner) are not requested again; owners
that were @-mentioned before (by anyone) are not pinged again. The pull request
author is skipped.

CODEOWNERS is read from the tip of the pull request's base branch, as GitHub
does. The script runs in .github/workflows/codeowners.yml on `workflow_run`,
from the default branch, and finds the pull request from the head repository,
branch and commit of the triggering run. It never checks out or executes code
of the pull request. File names of the pull request are not copied into the
comment, only CODEOWNERS rules.

Local dry run, without any write. With a token that has push access to the
repository, it also predicts who would be requested and who would be pinged:
    GITHUB_TOKEN=$(gh auth token) .github/workflows/scripts/codeowners_reviewers.py \\
        --repo BLAST-WarpX/warpx --pr 1234 --dry-run [--codeowners .github/CODEOWNERS]
Instead of --pr, the pull request can be given by its head, as in the workflow:
    --head-repo user/warpx --head-branch topic-x --head-sha <full commit hash>

Check the local CODEOWNERS (run in CI): every rule matches a tracked file and
applies to at least one (is not overridden by later rules by mistake), and every
owner is valid; with GITHUB_TOKEN set, also that every @user exists on GitHub:
    GITHUB_TOKEN=$(gh auth token) .github/workflows/scripts/codeowners_reviewers.py --check
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

API_URL = os.environ.get("GITHUB_API_URL", "https://api.github.com")
# GitHub uses the first CODEOWNERS file it finds, in this order.
CODEOWNERS_LOCATIONS = (".github/CODEOWNERS", "CODEOWNERS", "docs/CODEOWNERS")
# The account the workflow comments as, and a phrase of that comment: used to
# comment at most once per pull request.
BOT = os.environ.get("CODEOWNERS_BOT", "github-actions[bot]")
COMMENT_SIGNATURE = "You are pinged because you are listed as a code owner"
# Pull requests marked as work in progress, see CONTRIBUTING.md.
WIP = re.compile(r"^\s*\[WIP\]", re.IGNORECASE)
TIMEOUT = 30  # seconds per API request
RETRIES = 3


# --- GitHub REST API --------------------------------------------------------


class ApiError(Exception):
    def __init__(self, status, method, url, body):
        super().__init__(f"{method} {url} -> HTTP {status}: {body}")
        self.status = status
        self.body = body


def api(method, url, data=None, idempotent=True):
    """Call the GitHub REST API; return (decoded JSON body, response headers).

    Rate limits (HTTP 429, or 403 with a rate-limit message) are retried, since
    GitHub did not process the request. Server errors and network failures are
    only retried for idempotent requests: a retried comment could be posted twice.
    """
    if url.startswith("/"):
        url = API_URL + url
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "codeowners-reviewers-script",
    }
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    attempt = 0  # every iteration returns, raises, or retries after a wait
    while True:
        request = urllib.request.Request(
            url,
            method=method,
            data=None if data is None else json.dumps(data).encode(),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                body = response.read()
                return (json.loads(body) if body else None), response.headers
        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="replace")
            rate_limited = error.code == 429 or (
                error.code == 403 and "rate limit" in body.lower()
            )
            if attempt == RETRIES or not (
                rate_limited or (idempotent and error.code >= 500)
            ):
                raise ApiError(error.code, method, url, body) from None
            retry_after = error.headers.get("Retry-After") or ""
            wait = int(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1)
        except OSError:  # network errors and timeouts
            if attempt == RETRIES or not idempotent:
                raise
            wait = 2 ** (attempt + 1)
        print(f"Retrying {method} {url} in {wait} s", file=sys.stderr)
        time.sleep(min(wait, 60))
        attempt += 1


def api_list(path):
    """GET all pages of a list endpoint."""
    url = f"{API_URL}{path}{'&' if '?' in path else '?'}per_page=100"
    items = []
    while url:
        page, headers = api("GET", url)
        items.extend(page)
        match = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link") or "")
        url = match.group(1) if match else None
    return items


# --- CODEOWNERS -------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    line: int
    pattern: str
    owners: tuple
    regex: re.Pattern = field(compare=False, hash=False)


def pattern_to_regex(pattern):
    """Translate a CODEOWNERS pattern into a regex for repository file paths.

    CODEOWNERS uses gitignore patterns, without `!` negation, `[ ]` ranges and
    `\\#` escapes (not supported by GitHub either):
    - a leading `/`, or a `/` inside the pattern, anchors it at the repository
      root; otherwise it matches at any depth,
    - a trailing `/` matches everything inside that directory,
    - `**` matches across directories, `*` and `?` within one path component,
    - a pattern without wildcards in its last component matches a file of that
      name or a directory and everything inside it.
    GitHub documents that `docs/*` matches files in `docs/` but not in its
    subdirectories, so a last component with wildcards matches files only.
    """
    anchored = pattern.startswith("/") or "/" in pattern.strip("/")
    body = pattern.strip("/")
    out = []
    i = 0
    while i < len(body):
        if body.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif body.startswith("/**", i) and i + 3 == len(body):
            out.append("/.*")
            i += 3
        elif body.startswith("**", i):
            out.append(".*")
            i += 2
        elif body[i] == "*":
            out.append("[^/]*")
            i += 1
        elif body[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(body[i]))
            i += 1
    prefix = "" if anchored else "(?:.*/)?"
    if pattern.endswith("/"):
        suffix = "/.*"
    elif any(c in body.rsplit("/", 1)[-1] for c in "*?"):
        suffix = ""
    else:
        suffix = "(?:/.*)?"
    return re.compile(f"^{prefix}{''.join(out)}{suffix}$")


def parse_codeowners(text):
    rules = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = re.sub(r"(^|\s)#.*$", "", line).strip()  # comments
        if not line:
            continue
        pattern, *owners = line.split()
        rules.append(Rule(number, pattern, tuple(owners), pattern_to_regex(pattern)))
    return rules


def owning_rule(rules, path):
    """The last matching rule wins, as in GitHub."""
    for rule in reversed(rules):
        if rule.regex.match(path):
            return rule
    return None


def fetch_codeowners(repo, ref):
    """Return (path, text) of the CODEOWNERS file that GitHub uses at a commit."""
    for path in CODEOWNERS_LOCATIONS:
        try:
            content, _ = api("GET", f"/repos/{repo}/contents/{path}?ref={ref}")
        except ApiError as error:
            if error.status == 404:
                continue
            raise
        if content.get("encoding") != "base64":
            raise RuntimeError(f"{path} at {ref}: unexpected encoding (file > 1 MB?)")
        return path, base64.b64decode(content["content"]).decode()
    return None, None


USER = re.compile(r"^@[A-Za-z0-9-]+$")
TEAM = re.compile(r"^@[A-Za-z0-9-]+/[A-Za-z0-9._-]+$")
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def check(path):
    """Check the local CODEOWNERS file; return the number of errors (0: valid).

    Errors: a rule that matches no tracked file; a rule that applies to no file
    because later rules override it (the last matching rule wins), unless the
    overriding rules are all inside the rule's directory, i.e., the rule is a
    fallback for new subdirectories; owners with invalid syntax; and, if
    GITHUB_TOKEN is set, @users that do not exist on GitHub.
    """
    path = path or next((p for p in CODEOWNERS_LOCATIONS if Path(p).is_file()), None)
    if path is None:
        print("No CODEOWNERS file found.")
        return 1
    rules = parse_codeowners(Path(path).read_text())
    files = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, check=True
    ).stdout.decode()
    files = [f for f in files.split("\0") if f]
    winner = {f: owning_rule(rules, f) for f in files}
    applies = set(winner.values())
    errors = []
    users = {}  # login -> first line
    for rule in rules:
        matched = [f for f in files if rule.regex.match(f)]
        if not matched:
            errors.append(f"{path}:{rule.line}: {rule.pattern} matches no tracked file")
        elif rule not in applies:
            overriding = {winner[f] for f in matched}
            mistakes = [
                w
                for w in overriding
                if not (
                    rule.pattern.endswith("/") and w.pattern.startswith(rule.pattern)
                )
            ]
            if mistakes:
                errors.append(
                    f"{path}:{rule.line}: {rule.pattern} never applies, later rules"
                    " override it: "
                    + ", ".join(
                        f"{w.pattern} (line {w.line})"
                        for w in sorted(mistakes, key=lambda w: w.line)
                    )
                )
        for owner in rule.owners:
            if USER.match(owner):
                users.setdefault(owner[1:], rule.line)
            elif not (TEAM.match(owner) or EMAIL.match(owner)):
                errors.append(f"{path}:{rule.line}: invalid owner {owner}")
    if os.environ.get("GITHUB_TOKEN"):
        for login, line in sorted(users.items(), key=lambda kv: kv[0].lower()):
            try:
                user, _ = api("GET", f"/users/{login}")
            except ApiError as error:
                if error.status != 404:
                    raise
                errors.append(f"{path}:{line}: @{login} does not exist on GitHub")
                continue
            if user.get("type") != "User":
                errors.append(
                    f"{path}:{line}: @{login} is a {user.get('type')}, not a user"
                )
    else:
        print("GITHUB_TOKEN is not set: not checking that the @users exist on GitHub.")
    for error in errors:
        print(error)
    print(
        f"{path}: {len(rules)} rules, {len(users)} users checked, {len(errors)} error(s)"
    )
    return 1 if errors else 0


# --- pull requests ----------------------------------------------------------


def pending_review_requests(repo, number):
    """Pending review requests of people and teams: [(name, as_code_owner)].

    Team names are `org/slug`. Requests of bots (e.g., Copilot) are left out.
    Only the GraphQL API tells code owner requests apart (`asCodeOwner`).
    """
    org, name = repo.split("/")
    query = """query($owner: String!, $name: String!, $number: Int!) {
      repository(owner: $owner, name: $name) { pullRequest(number: $number) {
        reviewRequests(first: 100) { nodes { asCodeOwner requestedReviewer {
          __typename ... on User { login } ... on Team { slug } } } } } } }"""
    result, _ = api(
        "POST",
        "/graphql",
        {"query": query, "variables": {"owner": org, "name": name, "number": number}},
    )
    if result.get("errors"):
        raise RuntimeError(f"GraphQL errors: {result['errors']}")
    pending = []
    for node in result["data"]["repository"]["pullRequest"]["reviewRequests"]["nodes"]:
        reviewer = node["requestedReviewer"] or {}
        if reviewer.get("__typename") == "User":
            pending.append((reviewer["login"], node["asCodeOwner"]))
        elif reviewer.get("__typename") == "Team":
            pending.append((f"{org}/{reviewer['slug']}", node["asCodeOwner"]))
    return pending


def find_prs(repo, head_repo, head_branch, head_sha):
    """Return the numbers of the open pull requests whose head is this commit.

    A `workflow_run` event lists its pull requests only for branches of the
    same repository, not for forks, so look them up by head branch. Keep only
    pull requests whose head is still the commit that the triggering workflow
    ran on: after a newer push, its own run handles the pull request.
    """
    owner = head_repo.split("/")[0]
    head = urllib.parse.quote(f"{owner}:{head_branch}", safe="")
    return [
        p["number"]
        for p in api_list(f"/repos/{repo}/pulls?state=open&head={head}")
        if p["head"]["sha"] == head_sha
        and ((p["head"].get("repo") or {}).get("full_name") or "").lower()
        == head_repo.lower()
    ]


def process_pr(repo, number, args):
    """Request reviews from, or ping, the code owners of one pull request."""
    org = repo.split("/")[0]

    pr, _ = api("GET", f"/repos/{repo}/pulls/{number}")
    if pr["draft"]:
        print("Draft pull request: nothing to do.")
        return 0
    if WIP.match(pr["title"]):
        print("Pull request title starts with [WIP]: nothing to do.")
        return 0
    author = pr["user"]["login"].lower()

    # The pull request's history, read before CODEOWNERS and the changed files,
    # so that runs after the first one end early: who was ever requested (also
    # if the request was removed again) and by whom, who was ever @-mentioned
    # (GitHub adds a `mentioned` event for every @-mention in the description
    # or a comment, also for the pings of this script), and our own comment.
    requested_before, mentioned_before, requested_by_us = set(), set(), set()
    for event in api_list(f"/repos/{repo}/issues/{number}/timeline"):
        kind = event.get("event")
        if kind == "review_requested":
            if event.get("requested_reviewer"):
                name = event["requested_reviewer"]["login"]
            elif event.get("requested_team"):
                name = f"{org}/{event['requested_team']['slug']}"
            else:
                continue
            requested_before.add(name.lower())
            if (event.get("actor") or {}).get("login") == BOT:
                requested_by_us.add(name.lower())
        elif kind == "mentioned" and event.get("actor"):
            mentioned_before.add(event["actor"]["login"].lower())
        elif (
            kind == "commented"
            and (event.get("user") or {}).get("login") == BOT
            and COMMENT_SIGNATURE in (event.get("body") or "")
        ):
            print("Code owners were pinged before: nothing to do.")
            return 0

    # Reviews already organized: requests that a person added (not GitHub for
    # CODEOWNERS, not this script), or reviews of people other than the author.
    pending = pending_review_requests(repo, number)
    manual = [
        name
        for name, as_code_owner in pending
        if not as_code_owner and name.lower() not in requested_by_us
    ]
    if manual:
        print(f"Reviews already requested from {', '.join(manual)}: nothing to do.")
        return 0
    reviewers = {
        r["user"]["login"]
        for r in api_list(f"/repos/{repo}/pulls/{number}/reviews")
        if r.get("user")
        and r["user"].get("type") != "Bot"
        and r["user"]["login"].lower() != author
    }
    if reviewers:
        print(f"Already reviewed by {', '.join(sorted(reviewers))}: nothing to do.")
        return 0

    base_branch = pr["base"]["ref"]
    branch, _ = api(
        "GET", f"/repos/{repo}/branches/{urllib.parse.quote(base_branch, safe='')}"
    )
    base_sha = branch["commit"]["sha"]
    codeowners, text = fetch_codeowners(repo, base_sha)
    if args.codeowners:
        text = Path(args.codeowners).read_text()
        codeowners = codeowners or args.codeowners
        print(f"Using local {args.codeowners}; comment links refer to {base_branch}.")
    if text is None:
        print(f"No CODEOWNERS file on {base_branch}: nothing to do.")
        return 0
    rules = parse_codeowners(text)

    # Owners of the changed files: owner -> {rule: number of files}.
    # Renamed files count for their old and new location.
    paths = set()
    for f in api_list(f"/repos/{repo}/pulls/{number}/files"):
        paths.add(f["filename"])
        if f.get("previous_filename"):
            paths.add(f["previous_filename"])
    owners = {}
    for path in sorted(paths):
        rule = owning_rule(rules, path)
        for owner in rule.owners if rule else ():
            if not owner.startswith("@"):
                print(f"Skipping owner {owner} (e-mail): cannot request or ping.")
                continue
            files = owners.setdefault(owner[1:], {})
            files[rule] = files.get(rule, 0) + 1

    # Not requested again: the author, owners with a pending request (e.g., by
    # GitHub for CODEOWNERS), and owners requested before.
    skip = requested_before | {name.lower() for name, _ in pending} | {author}
    by_name = sorted(owners, key=str.lower)
    candidates = [o for o in by_name if o.lower() not in skip]

    # Request reviews one by one: one non-member would fail a whole batch.
    # Non-members are pinged instead, unless they were @-mentioned before.
    requested, ping, mentioned, failed, notes = [], [], [], [], []
    for owner in candidates:
        if "/" in owner:  # @org/team
            data = {"team_reviewers": [owner.split("/", 1)[1]]}
        else:
            data = {"reviewers": [owner]}
        member = None
        if args.dry_run:
            # predict the outcome; needs a token with push access to the repo
            try:
                if "/" not in owner:
                    api("GET", f"/repos/{repo}/collaborators/{owner}")
                member = True
            except ApiError as error:
                if error.status != 404:
                    raise
                member = False
        else:
            try:
                api("POST", f"/repos/{repo}/pulls/{number}/requested_reviewers", data)
                member = True
            except ApiError as error:
                # GitHub: "Reviews may only be requested from collaborators. One or
                # more of the users or teams you specified is not a collaborator ..."
                if error.status == 422 and "not a collaborator" in error.body.lower():
                    member = False
                else:
                    notes.append(f"- review request for `{owner}` failed: {error}")
            except OSError as error:
                notes.append(f"- review request for `{owner}` failed: {error}")
        if member is None:
            failed.append(owner)
        elif member:
            requested.append(owner)
        elif owner.lower() in mentioned_before:
            mentioned.append(owner)
        else:
            ping.append(owner)

    if ping:
        base = f"https://github.com/{repo}/blob/{base_sha}/{codeowners}"
        reasons = []
        for owner in ping:
            rules_text = ", ".join(
                f"[`{rule.pattern}`]({base}#L{rule.line})"
                f" ({count} file{'s' if count > 1 else ''})"
                for rule, count in sorted(
                    owners[owner].items(), key=lambda kv: kv[0].line
                )
            )
            reasons.append(f"@{owner}: {rules_text}")
        body = (
            " ".join(f"@{o}" for o in ping)
            + ": this pull request changes files that you look after."
            " A review or comment from you would be very welcome!\n\n"
            f"<sub>{COMMENT_SIGNATURE} in"
            f" [`{codeowners}`]({base}) for the following rules."
            " GitHub can only request reviews from members of the"
            f" {org} GitHub organization, so this comment pings you instead.<br>"
            + "<br>".join(reasons)
            + "</sub>\n"
        )
        if args.dry_run:
            print(f"[dry run] would comment:\n{body}")
        else:
            try:
                api(
                    "POST",
                    f"/repos/{repo}/issues/{number}/comments",
                    {"body": body},
                    idempotent=False,
                )
            except (ApiError, OSError) as error:
                failed.extend(ping)
                notes.append(f"- comment pinging {', '.join(ping)} failed: {error}")
                ping = []

    summary = [
        f"### Code owners of #{number}" + (" (dry run)" if args.dry_run else ""),
        f"- owners of changed files: {', '.join(by_name) or 'none'}",
        "- handled before or author: "
        + (", ".join(o for o in by_name if o not in candidates) or "none"),
        f"- review requested: {', '.join(requested) or 'none'}",
        f"- pinged in a comment: {', '.join(ping) or 'none'}",
        f"- not pinged, @-mentioned before: {', '.join(mentioned) or 'none'}",
        f"- failed: {', '.join(failed) or 'none'}",
    ] + notes
    print("\n".join(summary))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("\n".join(summary) + "\n")
    return 1 if failed else 0


# --- main -------------------------------------------------------------------


def main():
    env = os.environ.get
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=env("GITHUB_REPOSITORY"))
    parser.add_argument("--pr", type=int, default=env("PR_NUMBER"))
    parser.add_argument("--head-repo", default=env("HEAD_REPO"))
    parser.add_argument("--head-branch", default=env("HEAD_BRANCH"))
    parser.add_argument("--head-sha", default=env("HEAD_SHA"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=env("DRY_RUN") == "1",
        help="only print what would be requested and commented",
    )
    parser.add_argument(
        "--codeowners",
        metavar="FILE",
        help="read this local CODEOWNERS file instead of the one on the base branch"
        " (for testing; comment links still point to the base branch)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check the local CODEOWNERS file (or --codeowners FILE) and exit",
    )
    args = parser.parse_args()
    if args.check:
        return check(args.codeowners)
    if not args.repo:
        parser.error("set --repo or GITHUB_REPOSITORY")
    if not (args.pr or (args.head_repo and args.head_branch and args.head_sha)):
        parser.error("set --pr, or --head-repo, --head-branch and --head-sha")
    if args.pr:
        numbers = [args.pr]
    else:
        numbers = find_prs(args.repo, args.head_repo, args.head_branch, args.head_sha)
        if not numbers:
            print(
                f"No open pull request with head {args.head_repo}:{args.head_branch}"
                f" at {args.head_sha}: nothing to do."
            )
    return max((process_pr(args.repo, n, args) for n in numbers), default=0)


if __name__ == "__main__":
    sys.exit(main())
