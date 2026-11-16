"""Package declared model sources from a fixed official workflow-template checkout."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def declarations(value: object) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict) and isinstance(properties.get("models"), list):
            for model in properties["models"]:
                if not isinstance(model, dict) or not all(
                    isinstance(model.get(key), str) for key in ("name", "directory", "url")
                ):
                    raise ValueError("template model declaration lacks name, directory or URL")
                row = {key: model[key] for key in ("name", "directory", "url")}
                parsed = urlsplit(row["url"])
                if (
                    parsed.scheme == "https"
                    and parsed.netloc == "huggingface.co"
                    and parsed.query == "download=true"
                    and not parsed.fragment
                ):
                    row["url"] = urlunsplit(parsed._replace(query=""))
                for key in ("hash", "hash_type"):
                    if key in model:
                        if not isinstance(model[key], str):
                            raise ValueError("template model hash declaration must be text")
                        row[key] = model[key]
                found.append(row)
        for child in value.values():
            found.extend(declarations(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(declarations(child))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src/dinkster/template-model-sources.json",
    )
    args = parser.parse_args()
    head = subprocess.check_output(
        ["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != args.revision:
        parser.error("template checkout HEAD differs from the requested revision")
    if subprocess.check_output(
        ["git", "-C", str(args.checkout), "status", "--porcelain"], text=True
    ).strip():
        parser.error("template checkout must be clean")
    rows: dict[str, dict[str, str]] = {}
    templates = sorted((args.checkout / "templates").glob("*.json"))
    if not templates:
        parser.error("template checkout has no templates/*.json files")
    for path in templates:
        for row in declarations(json.loads(path.read_text(encoding="utf-8"))):
            rows[json.dumps(row, sort_keys=True)] = row
    args.output.write_text(
        json.dumps(
            {
                "repository": "https://github.com/Comfy-Org/workflow_templates",
                "revision": head,
                "models": [rows[key] for key in sorted(rows)],
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
