"""Mirror a GitHub-published engine and expose its channel only after verification."""

from __future__ import annotations

import argparse
import json
import os
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
from dinkster_supervisor.environment_channels import write_channel
from dinkster_supervisor.environment_downloads import DownloadError, fetch_artifact, read_metadata
from dinkster_supervisor.environment_state import Channel, content_digest


def publish(release: Path, bases: list[Path], channel: str, minimum_desktop_version: str) -> None:
    engine = json.loads((release / "release-manifest.json").read_bytes())
    expected_ref = "refs/heads/main" if channel == "main" else f"refs/tags/{engine['tag']}"
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
            github + quote(engine["tag"], safe="") + "/release-manifest.json", timeout=120
        ) as response:
            if response.read() != (release / "release-manifest.json").read_bytes():
                raise ValueError("GitHub engine manifest differs from publication input")
        for path in bases:
            base = json.loads(path.read_bytes())
            location = f"base/{base['variant']}/{base['id'][7:]}/manifest.json"
            if read_metadata(public, location) != path.read_bytes():
                raise ValueError(f"public base manifest differs: {base['variant']}")
        client = boto3.client(
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

        identity = content_digest(release / "release-manifest.json")[7:]
        prefix = f"engine/{identity}"
        rows = [
            *engine["artifacts"],
            *(
                {
                    "name": name,
                    "size": (release / name).stat().st_size,
                    "sha256": content_digest(release / name)[7:],
                }
                for name in ("release-manifest.json", "SHA256SUMS")
            ),
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
        path = work / "channels" / f"{channel}.json"
        if channel == "main":
            try:
                published_bytes = read_metadata(public, "channels/main.json")
            except DownloadError as error:
                cause = error.__cause__
                if not isinstance(cause, HTTPError) or cause.code != 404:
                    raise
            else:
                published = Channel.parse(parse_metadata(published_bytes))
                manifest = validate_manifest(
                    read_metadata(
                        public, f"engine/{published.engine_release[7:]}/release-manifest.json"
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
    parser.add_argument("--base-manifest", type=Path, action="append", required=True)
    parser.add_argument("--channel", choices=("main", "stable"), required=True)
    parser.add_argument("--minimum-desktop-version", required=True)
    args = parser.parse_args()
    publish(args.release, args.base_manifest, args.channel, args.minimum_desktop_version)
