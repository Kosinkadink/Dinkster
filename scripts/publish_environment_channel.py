"""Mirror a GitHub-published engine and expose its channel only after verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

import boto3
from boto3.exceptions import S3UploadFailedError
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    EndpointConnectionError,
    ReadTimeoutError,
)
from dinkster_supervisor.environment_artifacts import parse_metadata, validate_manifest
from dinkster_supervisor.environment_channels import engine_publication_names, write_channel
from dinkster_supervisor.environment_downloads import fetch_artifact, read_metadata
from dinkster_supervisor.environment_state import Channel, content_digest


def object_store():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(
            retries={"mode": "standard", "total_max_attempts": 4},
            connect_timeout=120,
            read_timeout=120,
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def inventory(client) -> tuple[dict[str, int], list[tuple[str, str, dict]]]:
    objects = {
        row["Key"]: row["Size"]
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=os.environ["R2_BUCKET"], Prefix="engine/"
        )
        for row in page.get("Contents", [])
    }
    manifests = []
    for key in sorted(objects):
        if not key.endswith("/release-manifest.json") or len(key.split("/")) != 3:
            continue
        data = client.get_object(Bucket=os.environ["R2_BUCKET"], Key=key)["Body"].read()
        identity = "sha256:" + hashlib.sha256(data).hexdigest()
        engine = validate_manifest(data, identity, None)
        name, _ = engine_publication_names(engine, identity, "main")
        if key != f"engine/{name}/release-manifest.json":
            raise ValueError(f"engine directory differs from its manifest identity: {key}")
        manifests.append((key, identity, engine))
    return objects, manifests


def check_collision(client, identity: str) -> None:
    objects, manifests = inventory(client)
    known = {key: full_id for key, full_id, _ in manifests}
    for directory in {key.split("/")[1] for key in objects if len(key.split("/")) >= 3}:
        short = (
            directory[:8]
            if re.fullmatch(r"[0-9a-f]{64}", directory)
            else directory.rsplit("-", 1)[-1]
        )
        if short != identity[7:15]:
            continue
        existing = known.get(f"engine/{directory}/release-manifest.json")
        if existing is None or existing != identity:
            raise ValueError(f"engine hash8 collision or unidentifiable directory: {directory}")


def write_index(client, output: Path) -> None:
    objects, manifests = inventory(client)
    rows = []
    for key, identity, engine in manifests:
        prefix = key.rsplit("/", 1)[0]
        if f"{prefix}/SHA256SUMS" not in objects or any(
            objects.get(f"{prefix}/{row['name']}") != row["size"] for row in engine["artifacts"]
        ):
            continue
        names = [
            key,
            f"{prefix}/SHA256SUMS",
            *(f"{prefix}/{r['name']}" for r in engine["artifacts"]),
        ]
        rows.append(
            {
                "label": engine.get("label"),
                "sha256": identity[7:],
                "commit": engine["commit"],
                "version": engine["version"],
                "builtAt": engine.get("builtAt"),
                "size": sum(objects[name] for name in names),
                "path": prefix,
            }
        )
    rows.sort(key=lambda row: (row["builtAt"] or "", row["sha256"]), reverse=True)
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def publish(release: Path, bases: list[Path], channel: str, minimum_desktop_version: str) -> None:
    engine = json.loads((release / "release-manifest.json").read_bytes())
    identity = content_digest(release / "release-manifest.json")
    name, tag = engine_publication_names(engine, identity, channel)
    expected_ref = "refs/heads/main" if channel == "main" else f"refs/tags/{tag}"
    if (
        os.environ.get("GITHUB_REPOSITORY") != "Kosinkadink/Dinkster"
        or os.environ.get("GITHUB_EVENT_NAME") != "push"
        or os.environ.get("GITHUB_REF") != expected_ref
        or os.environ.get("GITHUB_SHA") != engine["commit"]
    ):
        raise ValueError("channel publication requires the exact repository push")
    if not all(
        os.environ.get(name)
        for name in (
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_ACCOUNT_ID",
            "R2_BUCKET",
            "R2_PUBLIC_BASE_URL",
        )
    ):
        print("R2 channel publication skipped: credentials or configuration are absent")
        return
    public = os.environ["R2_PUBLIC_BASE_URL"].rstrip("/")
    with tempfile.TemporaryDirectory(prefix="engine-channel-") as directory:
        work = Path(directory)
        write_channel(
            release,
            bases,
            channel=channel,
            output_root=work,
            minimum_desktop_version=minimum_desktop_version,
        )
        github = "https://github.com/Kosinkadink/Dinkster/releases/download/"
        with urlopen(
            github + quote(tag, safe="") + "/release-manifest.json", timeout=120
        ) as response:
            if response.read() != (release / "release-manifest.json").read_bytes():
                raise ValueError("GitHub engine manifest differs from publication input")
        for path in bases:
            base = json.loads(path.read_bytes())
            base_name = base.get("label", base["id"][7:])
            location = f"base/{base['variant']}/{base_name}/manifest.json"
            if read_metadata(public, location) != path.read_bytes():
                raise ValueError(f"public base manifest differs: {base['variant']}")
        client = object_store()
        check_collision(client, identity)
        transfer = TransferConfig(
            multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=64 * 1024 * 1024,
            max_concurrency=1,
            use_threads=False,
            preferred_transfer_client="classic",
        )

        def upload(path: Path, key: str, *, extra: dict[str, str] | None = None) -> None:
            for attempt in range(4):
                try:
                    client.upload_file(
                        str(path), os.environ["R2_BUCKET"], key, ExtraArgs=extra, Config=transfer
                    )
                    return
                except (
                    S3UploadFailedError,
                    ClientError,
                    ConnectionClosedError,
                    EndpointConnectionError,
                    ReadTimeoutError,
                ) as error:
                    cause = (
                        error.__cause__ or error.__context__
                        if isinstance(error, S3UploadFailedError)
                        else error
                    )
                    transient = isinstance(
                        cause, (ConnectionClosedError, EndpointConnectionError, ReadTimeoutError)
                    ) or (
                        isinstance(cause, ClientError)
                        and (
                            cause.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                            in {429, 500, 502, 503, 504}
                            or cause.response.get("Error", {}).get("Code")
                            in {"ServiceUnavailable", "SlowDown", "RequestTimeout", "InternalError"}
                        )
                    )
                    if not transient or attempt == 3:
                        raise
                    delay = 2**attempt
                    print(
                        f"Transient R2 upload failure for {key}; retrying in {delay}s", flush=True
                    )
                    time.sleep(delay)

        prefix = f"engine/{name}"
        rows = [
            *(
                {
                    "name": name,
                    "size": (release / name).stat().st_size,
                    "sha256": content_digest(release / name)[7:],
                }
                for name in ("release-manifest.json", "SHA256SUMS")
            ),
            *engine["artifacts"],
        ]
        for row in rows:
            key = f"{prefix}/{row['name']}"
            try:
                client.head_object(Bucket=os.environ["R2_BUCKET"], Key=key)
            except ClientError as error:
                if error.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
                    raise
            else:
                continue
            upload(release / row["name"], key)
        for row in rows:
            fetch_artifact(
                (f"{public}/{prefix}",),
                row["name"],
                work / "verified" / row["name"],
                row["sha256"],
                row["size"],
            )
        index = work / "engine-index.json"
        write_index(client, index)
        upload(
            index,
            "engine/index.json",
            extra={"ContentType": "application/json", "CacheControl": "no-cache"},
        )
        if read_metadata(public, "engine/index.json") != index.read_bytes():
            raise ValueError("public engine index differs after publication")
        path = work / "channels" / f"{channel}.json"
        if channel == "main":
            try:
                published_bytes = read_metadata(public, "channels/main.json")
            except HTTPError as error:
                if error.code != 404:
                    raise
            else:
                published = Channel.parse(parse_metadata(published_bytes))
                manifest = validate_manifest(
                    read_metadata(
                        public,
                        next(
                            location.path
                            for location in published.engine_locations
                            if location.kind == "mirror"
                        ),
                    ),
                    published.engine_release,
                    None,
                )
                published_commit = manifest["commit"]
                with urlopen(
                    "https://api.github.com/repos/Kosinkadink/Dinkster/compare/"
                    f"{published_commit}...{engine['commit']}",
                    timeout=120,
                ) as response:
                    status = json.load(response)["status"]
                if status == "identical":
                    if published_bytes != path.read_bytes():
                        raise ValueError("public channel differs for the same engine commit")
                    print(f"Verified main channel for engine {identity}")
                    return
                if status in {"behind", "diverged"}:
                    print(
                        f"Main channel skipped: published {published_commit}, "
                        f"candidate {engine['commit']} ({status})"
                    )
                    return
                if status != "ahead":
                    raise ValueError(f"unexpected GitHub comparison status: {status}")
        upload(
            path,
            f"channels/{channel}.json",
            extra={"ContentType": "application/json", "CacheControl": "no-cache"},
        )
        if read_metadata(public, f"channels/{channel}.json") != path.read_bytes():
            raise ValueError("public channel differs after publication")
        print(f"Published {channel} channel for engine {identity}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--base-manifest", type=Path, action="append")
    parser.add_argument("--channel", choices=("main", "stable"))
    parser.add_argument("--minimum-desktop-version")
    args = parser.parse_args()
    if args.preflight:
        check_collision(object_store(), content_digest(args.release / "release-manifest.json"))
    elif not args.base_manifest or not args.channel or not args.minimum_desktop_version:
        parser.error("publication requires base manifests, channel and minimum Desktop version")
    else:
        publish(args.release, args.base_manifest, args.channel, args.minimum_desktop_version)
