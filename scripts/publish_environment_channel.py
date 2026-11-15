"""Mirror a GitHub-published engine and expose its channel only after verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

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
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from dinkster_supervisor.environment_artifacts import (
    artifact_rows,
    parse_metadata,
    validate_manifest,
)
from dinkster_supervisor.environment_channel_signatures import sign_channel, verify_channel
from dinkster_supervisor.environment_channels import engine_publication_names, write_channel
from dinkster_supervisor.environment_downloads import DownloadError, fetch_artifact, read_metadata
from dinkster_supervisor.environment_frontend import (
    frontend_filename,
    select_frontend,
    verify_frontend_wheel,
)
from dinkster_supervisor.environment_state import Channel, content_digest, write_document
from dinkster_supervisor.environment_tools import tools_reference, validate_tools_manifest
from packaging.specifiers import SpecifierSet


def sign_channel_file(path: Path, channel: str) -> None:
    """Sign final selections and refuse an unpinned workflow key before upload."""
    started = time.perf_counter()
    key = load_pem_private_key(
        os.environ["DINKSTER_CHANNEL_SIGNING_KEY"].encode("ascii"), password=None
    )
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("channel publication requires an Ed25519 private key")
    envelope = sign_channel(parse_metadata(path.read_bytes()), channel, key)
    verify_channel(envelope, channel)
    path.write_bytes(envelope)
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"Channel {channel} signing and self-verification: {elapsed_ms:.5f} ms")


def initialize_nightly(client, stable_path: Path, public: str) -> None:
    """Initialize Nightly solely from authenticated Stable selections."""
    stable = verify_channel(stable_path.read_bytes(), "stable")
    Channel.parse(stable)
    try:
        previous = read_metadata(public, "channels/nightly.json")
    except DownloadError as error:
        cause = error.__cause__
        if not isinstance(cause, HTTPError) or cause.code != 404:
            raise
    else:
        document = parse_metadata(previous)
        if "signed" in document or "signatures" in document:
            Channel.parse(verify_channel(previous, "nightly"))
            client.delete_object(Bucket=os.environ["R2_BUCKET"], Key="channels/main.json")
            print("Signed Nightly already exists; initialization is unnecessary")
            return
        if (
            document.get("format") != "dinkster.channel/1"
            or document.get("signature") is not None
            or document.get("channel") not in {None, "nightly"}
        ):
            raise ValueError("Nightly initialization requires a missing or legacy unsigned channel")
    document = dict(stable, channel="nightly")
    document.pop("nightlyDate", None)
    Channel.parse(document)
    path = stable_path.with_name("nightly.json")
    write_document(path, document)
    sign_channel_file(path, "nightly")
    upload(
        client,
        path,
        "channels/nightly.json",
        extra={"ContentType": "application/json", "CacheControl": "no-cache"},
    )
    published = read_metadata(public, "channels/nightly.json")
    if published != path.read_bytes():
        raise ValueError("public Nightly differs after authenticated initialization")
    Channel.parse(verify_channel(published, "nightly"))
    client.delete_object(Bucket=os.environ["R2_BUCKET"], Key="channels/main.json")
    print("Initialized Nightly from authenticated Stable; no unsigned selections retained")


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


def inventory(client, scope: str = "engine") -> tuple[dict[str, int], list[tuple[str, str, dict]]]:
    manifest_name = "tools-manifest.json" if scope == "tools" else "release-manifest.json"
    objects = {
        row["Key"]: row["Size"]
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=os.environ["R2_BUCKET"], Prefix=scope + "/"
        )
        for row in page.get("Contents", [])
    }
    manifests = []
    for key in sorted(objects):
        if not key.endswith("/" + manifest_name) or len(key.split("/")) != 3:
            continue
        data = client.get_object(Bucket=os.environ["R2_BUCKET"], Key=key)["Body"].read()
        identity = "sha256:" + hashlib.sha256(data).hexdigest()
        if scope == "tools":
            manifest = validate_tools_manifest(data, identity[7:])
            name = f"{manifest['label']}-{identity[7:15]}"
        else:
            manifest = validate_manifest(data, identity, None)
            name, _ = engine_publication_names(manifest, identity, "nightly")
        if key != f"{scope}/{name}/{manifest_name}":
            raise ValueError(f"{scope} directory differs from its manifest identity: {key}")
        manifests.append((key, identity, manifest))
    return objects, manifests


def check_collision(client, identity: str, scope: str = "engine") -> None:
    objects, manifests = inventory(client, scope)
    manifest_name = "tools-manifest.json" if scope == "tools" else "release-manifest.json"
    known = {key: full_id for key, full_id, _ in manifests}
    for directory in {key.split("/")[1] for key in objects if len(key.split("/")) >= 3}:
        short = (
            directory[:8]
            if re.fullmatch(r"[0-9a-f]{64}", directory)
            else directory.rsplit("-", 1)[-1]
        )
        if short != identity[7:15]:
            continue
        existing = known.get(f"{scope}/{directory}/{manifest_name}")
        if existing is None or existing != identity:
            raise ValueError(f"{scope} hash8 collision or unidentifiable directory: {directory}")


def write_index(client, output: Path, scope: str = "engine") -> None:
    objects, manifests = inventory(client, scope)
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
                "label": prefix.split("/")[1] if scope == "tools" else engine.get("label"),
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


def upload(client, path: Path, key: str, *, extra: dict[str, str] | None = None) -> None:
    """Use the same bounded transfer/retry policy for engine and tools objects."""
    transfer = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=1,
        use_threads=False,
        preferred_transfer_client="classic",
    )
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
            print(f"Transient R2 upload failure for {key}; retrying in {delay}s", flush=True)
            time.sleep(delay)


def mirror_engine(client, release: Path, work: Path) -> None:
    engine = json.loads((release / "release-manifest.json").read_bytes())
    identity = content_digest(release / "release-manifest.json")
    name, _ = engine_publication_names(engine, identity, "nightly")
    public = os.environ["R2_PUBLIC_BASE_URL"].rstrip("/")
    check_collision(client, identity)
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
        upload(client, release / row["name"], key)
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
        client,
        index,
        "engine/index.json",
        extra={"ContentType": "application/json", "CacheControl": "no-cache"},
    )
    if read_metadata(public, "engine/index.json") != index.read_bytes():
        raise ValueError("public engine index differs after publication")


def recover_engine_mirrors(report_path: Path) -> None:
    if (
        os.environ.get("GITHUB_REPOSITORY") != "Kosinkadink/Dinkster"
        or os.environ.get("GITHUB_EVENT_NAME") != "push"
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
    ):
        raise ValueError("engine mirror recovery requires the owning repository main push")
    report = json.loads(report_path.read_bytes())
    pins = [
        pin
        for pin in report["unresolvedConsumerPins"]
        if pin["repository"] == "Kosinkadink/Dinkster"
        and pin.get("scope", "engine") == "engine"
        and ("tag" in pin or "engine" in pin)
    ]
    if not pins:
        print("No unresolved engine consumers require mirror recovery")
        return
    token = os.environ["GITHUB_TOKEN"]
    client = object_store()
    with tempfile.TemporaryDirectory(prefix="engine-mirror-recovery-") as directory:
        work = Path(directory)
        staged = []
        page = 1
        while pins:
            request = Request(
                "https://api.github.com/repos/Kosinkadink/Dinkster/releases"
                f"?per_page=100&page={page}",
                headers={"Authorization": f"Bearer {token}"},
            )
            with urlopen(request, timeout=120) as response:
                releases = json.load(response)
            if not releases:
                raise ValueError(f"No original GitHub engine release satisfies consumers: {pins}")
            for published in releases:
                tag = published["tag_name"]
                if published["draft"] or not re.fullmatch(r"(?:main-|v)[A-Za-z0-9._+-]+", tag):
                    continue
                if not any("engine" in pin or pin.get("tag") == tag for pin in pins):
                    continue
                assets = {row["name"]: row for row in published["assets"]}
                if "release-manifest.json" not in assets:
                    continue
                release = work / tag
                release.mkdir()
                source = "https://github.com/Kosinkadink/Dinkster/releases/download/" + quote(
                    tag, safe=""
                )
                manifest_asset = assets["release-manifest.json"]
                identity = manifest_asset.get("digest", "")
                if not re.fullmatch(r"sha256:[0-9a-f]{64}", identity):
                    raise ValueError(f"GitHub engine manifest digest is missing: {tag}")
                fetch_artifact(
                    (source,),
                    "release-manifest.json",
                    release / "release-manifest.json",
                    identity[7:],
                    manifest_asset["size"],
                )
                manifest = validate_manifest(
                    (release / "release-manifest.json").read_bytes(), identity, None
                )
                _, expected_tag = engine_publication_names(
                    manifest, identity, "nightly" if published["prerelease"] else "stable"
                )
                if manifest.get("repository") != "Kosinkadink/Dinkster" or tag != expected_tag:
                    raise ValueError(f"GitHub engine tag differs from its manifest identity: {tag}")
                matching = [
                    pin
                    for pin in pins
                    if ("tag" not in pin or pin["tag"] == tag)
                    and ("identity" not in pin or pin["identity"] == identity)
                    and (
                        "engine" not in pin
                        or SpecifierSet(pin["engine"]).contains(
                            manifest["version"], prereleases=True
                        )
                    )
                ]
                if not matching:
                    continue
                for row in [
                    *artifact_rows(manifest["artifacts"], "name"),
                    {
                        "name": "SHA256SUMS",
                        "sha256": assets["SHA256SUMS"]["digest"][7:],
                        "size": assets["SHA256SUMS"]["size"],
                    },
                ]:
                    name = row["name"]
                    asset = assets[name]
                    if (
                        asset.get("digest") != "sha256:" + row["sha256"]
                        or asset["size"] != row["size"]
                    ):
                        raise ValueError(
                            f"GitHub engine artifact differs from manifest: {tag}/{name}"
                        )
                    fetch_artifact((source,), name, release / name, row["sha256"], row["size"])
                staged.append(release)
                pins = [pin for pin in pins if pin not in matching]
                if not pins:
                    break
            page += 1
        for release in staged:
            mirror_engine(client, release, work / release.name)
            print(f"Recovered original engine mirror: {release.name}")


def publish(
    release: Path,
    bases: list[Path],
    channel: str,
    minimum_desktop_version: str,
    frontend: dict | None = None,
    tools_release: Path | None = None,
) -> None:
    engine = json.loads((release / "release-manifest.json").read_bytes())
    identity = content_digest(release / "release-manifest.json")
    _, tag = engine_publication_names(engine, identity, channel)
    expected_ref = "refs/heads/main" if channel == "nightly" else f"refs/tags/{tag}"
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
        if channel == "stable":
            raise ValueError("Stable publication requires R2 credentials and configuration")
        print("R2 channel publication skipped: credentials or configuration are absent")
        return
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("GITHUB_TOKEN is required for channel publication")
    public = os.environ["R2_PUBLIC_BASE_URL"].rstrip("/")
    with tempfile.TemporaryDirectory(prefix="engine-channel-") as directory:
        work = Path(directory)
        write_channel(
            release,
            bases,
            channel=channel,
            output_root=work,
            minimum_desktop_version=minimum_desktop_version,
            frontend=frontend,
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
        mirror_engine(client, release, work)
        path = work / "channels" / f"{channel}.json"
        document = parse_metadata(path.read_bytes())
        document.pop("signature", None)
        document["channel"] = channel
        write_document(path, document)
        if channel == "stable":
            if tools_release is None:
                raise ValueError("Stable publication requires tools from the same engine source")
            tools_data = (tools_release / "tools-manifest.json").read_bytes()
            tools_digest = hashlib.sha256(tools_data).hexdigest()
            tools_manifest = validate_tools_manifest(tools_data, tools_digest)
            if tools_manifest["commit"] != engine["commit"]:
                raise ValueError("Stable tools and engine must come from the same commit")
            label = f"{tools_manifest['label']}-{tools_digest[:8]}"
            tools_url = f"{public}/tools/{label}"
            if read_metadata(tools_url, "tools-manifest.json") != tools_data:
                raise ValueError("public Stable tools manifest differs from publication input")
            for row in artifact_rows(tools_manifest["artifacts"], "name"):
                fetch_artifact(
                    (tools_url,),
                    row["name"],
                    work / "verified-tools" / row["name"],
                    row["sha256"],
                    row["size"],
                )
            document = parse_metadata(path.read_bytes())
            document["tools"] = tools_reference(
                {
                    "label": label,
                    "sha256": tools_digest,
                    "manifest": f"{tools_url}/tools-manifest.json",
                    "release": tools_url,
                }
            )
            Channel.parse(document)
            write_document(path, document)
        if channel == "nightly":
            publication_date = datetime.now(UTC).date().isoformat()
            document = parse_metadata(path.read_bytes())
            document["nightlyDate"] = publication_date
            write_document(path, document)
            try:
                published_bytes = read_metadata(public, "channels/nightly.json")
            except DownloadError as error:
                cause = error.__cause__
                if not isinstance(cause, HTTPError) or cause.code != 404:
                    raise
            else:
                published_document = verify_channel(published_bytes, channel)
                published = Channel.parse(published_document)
                if frontend is None and "frontend" in published_document:
                    document = parse_metadata(path.read_bytes())
                    document["frontend"] = published_document["frontend"]
                    select_frontend(engine, document)
                    write_document(path, document)
                if "tools" in published_document:
                    tools = tools_reference(published_document["tools"])
                    tools_manifest = validate_tools_manifest(
                        read_metadata(tools["release"], "tools-manifest.json"), tools["sha256"]
                    )
                    if tools["label"] != f"{tools_manifest['label']}-{tools['sha256'][:8]}":
                        raise ValueError("published tools manifest identity differs from channel")
                    document = parse_metadata(path.read_bytes())
                    document["tools"] = tools
                    write_document(path, document)
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
                request = Request(
                    "https://api.github.com/repos/Kosinkadink/Dinkster/compare/"
                    f"{published_commit}...{engine['commit']}",
                    headers={"Authorization": f"Bearer {github_token}"},
                )
                with urlopen(request, timeout=120) as response:
                    status = json.load(response)["status"]
                if status == "identical":
                    document = parse_metadata(path.read_bytes())
                    if "nightlyDate" in published_document:
                        document["nightlyDate"] = published_document["nightlyDate"]
                    else:
                        document.pop("nightlyDate", None)
                    write_document(path, document)
                    if published_document == document:
                        client.delete_object(
                            Bucket=os.environ["R2_BUCKET"], Key="channels/main.json"
                        )
                        print(f"Verified Nightly channel for engine {identity}")
                        return
                    candidate = parse_metadata(path.read_bytes())
                    unchanged = dict(published_document)
                    candidate.pop("frontend", None)
                    unchanged.pop("frontend", None)
                    if frontend is None or candidate != unchanged:
                        raise ValueError("public channel differs for the same engine commit")
                if status in {"behind", "diverged"}:
                    print(
                        f"Nightly channel skipped: published {published_commit}, "
                        f"candidate {engine['commit']} ({status})"
                    )
                    return
                if status not in {"ahead", "identical"}:
                    raise ValueError(f"unexpected GitHub comparison status: {status}")
                previous_date = published_document.get("nightlyDate")
                if previous_date is not None:
                    if (
                        not isinstance(previous_date, str)
                        or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", previous_date) is None
                    ):
                        raise ValueError("published Nightly date must be YYYY-MM-DD")
                    if previous_date >= publication_date:
                        client.delete_object(
                            Bucket=os.environ["R2_BUCKET"], Key="channels/main.json"
                        )
                        print(f"Nightly channel already advanced on {previous_date}")
                        return
        selected_frontend = select_frontend(engine, parse_metadata(path.read_bytes()))
        if selected_frontend is not None:
            wheel_name = frontend_filename(selected_frontend)
            wheel_path = work / "verified" / wheel_name
            fetch_artifact(
                (cast("str", selected_frontend["url"]).rsplit("/", 1)[0],),
                wheel_name,
                wheel_path,
                cast("str", selected_frontend["sha256"]),
                cast("int", selected_frontend["size"]),
            )
            verify_frontend_wheel(wheel_path, selected_frontend)
        sign_channel_file(path, channel)
        upload(
            client,
            path,
            f"channels/{channel}.json",
            extra={"ContentType": "application/json", "CacheControl": "no-cache"},
        )
        if read_metadata(public, f"channels/{channel}.json") != path.read_bytes():
            raise ValueError("public channel differs after publication")
        if channel == "stable":
            Channel.parse(verify_channel(read_metadata(public, "channels/stable.json"), "stable"))
            initialize_nightly(client, path, public)
        if channel == "nightly":
            client.delete_object(Bucket=os.environ["R2_BUCKET"], Key="channels/main.json")
        print(f"Published {channel} channel for engine {identity}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path)
    parser.add_argument(
        "--recover-engine-mirrors", type=Path, help="fresh retention dry-run report"
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--base-manifest", type=Path, action="append")
    parser.add_argument("--channel", choices=("nightly", "stable"))
    parser.add_argument("--minimum-desktop-version")
    parser.add_argument("--frontend", type=Path, help="optional published frontend reference JSON")
    parser.add_argument("--tools", type=Path, help="same-source published tools release for Stable")
    args = parser.parse_args()
    if args.recover_engine_mirrors is not None:
        if (
            args.release
            or args.preflight
            or args.channel
            or args.base_manifest
            or args.tools
            or args.frontend
        ):
            parser.error("mirror recovery cannot select a release or change a channel")
        recover_engine_mirrors(args.recover_engine_mirrors)
    elif args.release is None:
        parser.error("publication requires a release directory")
    elif args.preflight:
        check_collision(object_store(), content_digest(args.release / "release-manifest.json"))
    elif not args.base_manifest or not args.channel or not args.minimum_desktop_version:
        parser.error("publication requires base manifests, channel and minimum Desktop version")
    else:
        publish(
            args.release,
            args.base_manifest,
            args.channel,
            args.minimum_desktop_version,
            parse_metadata(args.frontend.read_bytes()) if args.frontend is not None else None,
            args.tools,
        )
