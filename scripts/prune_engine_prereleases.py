"""Keep the newest 50 commit-named engine prereleases, never stable releases."""

from __future__ import annotations

import json
import os
import re
import subprocess


def prune(repository: str) -> list[int]:
    pages = json.loads(
        subprocess.check_output(
            ["gh", "api", "--paginate", "--slurp", f"repos/{repository}/releases?per_page=100"],
            text=True,
        )
    )
    releases = [
        release
        for page in pages
        for release in page
        if release["prerelease"]
        and not release["draft"]
        and re.fullmatch(r"main-[0-9a-f]{40}", release["tag_name"])
        and release["published_at"] is not None
    ]
    releases.sort(key=lambda release: (release["published_at"], release["id"]), reverse=True)
    removed = []
    for release in releases[50:]:
        path = f"repos/{repository}/releases/{release['id']}"
        current = json.loads(subprocess.check_output(["gh", "api", path], text=True))
        if (
            not current["prerelease"]
            or current["draft"]
            or current["tag_name"] != release["tag_name"]
        ):
            continue
        subprocess.run(["gh", "api", "--method", "DELETE", path], check=True)
        removed.append(release["id"])
    return removed


if __name__ == "__main__":
    print(json.dumps({"removedPrereleases": prune(os.environ["GITHUB_REPOSITORY"])}))
