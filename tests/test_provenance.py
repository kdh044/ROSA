from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

from rosa.provenance import capture_external_provenance


def _git(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", os.fspath(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def _initialize_repository(path: Path) -> None:
    path.mkdir(exist_ok=True)
    _git(path, "init", "--quiet", "--initial-branch=main")
    _git(path, "config", "user.name", "ROSA provenance test")
    _git(path, "config", "user.email", "rosa-provenance@example.invalid")


def test_external_provenance_is_deterministic_read_only_and_secret_safe(
    tmp_path: Path,
):
    source = tmp_path / "rosa"
    external = source / "external"
    dependency = tmp_path / "detector-host"
    work_dir = source / "results" / "experiment"
    external.mkdir(parents=True)
    _initialize_repository(source)
    _initialize_repository(dependency)

    source_file = source / "runner.py"
    source_file.write_text("SOURCE_VERSION = 1\n", encoding="utf-8")
    _git(source, "add", "runner.py")
    _git(source, "commit", "--quiet", "-m", "source baseline")
    source_file.write_text("SOURCE_VERSION = 2\n", encoding="utf-8")

    tracked = dependency / "model.py"
    binary = dependency / "lookup.bin"
    secret = dependency / ".env"
    target = dependency / "target.txt"
    tracked_link = dependency / "target-link"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    binary.write_bytes(b"\x00baseline-binary\xff")
    secret.write_text('API_KEY="baseline-secret-value"\n', encoding="utf-8")
    target.write_text("target\n", encoding="utf-8")
    tracked_link.symlink_to("target.txt")
    _git(dependency, "add", ".")
    _git(dependency, "commit", "--quiet", "-m", "baseline")

    tracked.write_text("VALUE = 2\n", encoding="utf-8")
    binary.write_bytes(b"\x00changed-binary\xfe\xfd")
    secret_value = b'API_KEY="do-not-copy-this-secret"\n'
    secret.write_bytes(secret_value)
    untracked_value = b"untracked private dataset payload\x00"
    untracked = dependency / "scratch.private"
    untracked.write_bytes(untracked_value)
    (external / "detector_host").symlink_to(dependency)

    status_before = _git(
        dependency,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    index_before = hashlib.sha256(
        (dependency / ".git" / "index").read_bytes()
    ).hexdigest()
    first = capture_external_provenance(source, work_dir)
    second = capture_external_provenance(source, work_dir)

    assert first == second
    expected_source_diff = _git(
        source,
        "diff",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        "--no-textconv",
        "HEAD",
        "--",
    )
    assert first["source_revision"] == {
        "branch": "main",
        "capture_attempts": 1,
        "head": _git(source, "rev-parse", "HEAD").decode().strip(),
        "stable_capture": True,
        "tracked_diff_sha256": hashlib.sha256(expected_source_diff).hexdigest(),
        "tracked_dirty": True,
    }
    assert (
        _git(
            dependency,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
        == status_before
    )
    assert (
        hashlib.sha256((dependency / ".git" / "index").read_bytes()).hexdigest()
        == index_before
    )

    manifest_path = work_dir / first["manifest"]
    manifest_bytes = manifest_path.read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == first["manifest_sha256"]
    manifest = json.loads(manifest_bytes)
    assert manifest["snapshot_sha256"] == first["snapshot_sha256"]
    assert manifest["source_revision"] == first["source_revision"]
    entry = manifest["external_entries"][0]
    assert entry["type"] == "symlink"
    assert entry["link_target"] == os.fspath(dependency)
    assert entry["resolved_path"] == os.fspath(dependency.resolve())

    repository = manifest["repositories"][0]
    assert repository["head"] == _git(dependency, "rev-parse", "HEAD").decode().strip()
    assert repository["dirty"] is True
    assert repository["stable_capture"] is True
    assert repository["tracked_symlinks"] == [
        {
            "link_target": "target.txt",
            "path": "target-link",
            "resolved_path": os.fspath(target.resolve()),
        }
    ]
    assert repository["omitted_tracked_diff_sections"][0]["path"] == ".env"
    assert repository["omitted_tracked_diff_sections"][0]["reason"] == "sensitive_path"

    snapshot_root = manifest_path.parent
    patch_metadata = repository["artifacts"]["tracked.patch"]
    patch = (snapshot_root / patch_metadata["path"]).read_bytes()
    assert b"+VALUE = 2" in patch
    assert b"GIT binary patch" in patch
    assert secret_value not in patch
    assert hashlib.sha256(patch).hexdigest() == patch_metadata["sha256"]

    untracked_metadata = repository["artifacts"]["untracked.json"]
    untracked_manifest = json.loads(
        (snapshot_root / untracked_metadata["path"]).read_bytes()
    )
    assert untracked_manifest == [
        {
            "path": "scratch.private",
            "sha256": hashlib.sha256(untracked_value).hexdigest(),
            "size": len(untracked_value),
            "type": "file",
        }
    ]

    for artifact in snapshot_root.rglob("*"):
        if artifact.is_file():
            contents = artifact.read_bytes()
            assert secret_value not in contents
            assert untracked_value not in contents


def test_non_repository_external_entry_is_recorded_without_copying(
    tmp_path: Path,
):
    source = tmp_path / "rosa"
    external = source / "external"
    dependency = tmp_path / "plain-dependency"
    work_dir = tmp_path / "experiment"
    external.mkdir(parents=True)
    _initialize_repository(source)
    (source / "README.md").write_text("clean source\n", encoding="utf-8")
    _git(source, "add", "README.md")
    _git(source, "commit", "--quiet", "-m", "clean source")
    dependency.mkdir()
    payload = b"not a Git repository and must not be copied"
    (dependency / "payload.dat").write_bytes(payload)
    (external / "plain").symlink_to(dependency)

    reference = capture_external_provenance(source, work_dir)
    manifest_path = work_dir / reference["manifest"]
    manifest = json.loads(manifest_path.read_bytes())

    assert manifest["external_entries"][0]["repository_id"] is None
    assert manifest["repositories"] == []
    assert reference["source_revision"]["tracked_dirty"] is False
    assert (
        reference["source_revision"]["tracked_diff_sha256"]
        == hashlib.sha256(b"").hexdigest()
    )
    assert all(
        payload not in artifact.read_bytes()
        for artifact in manifest_path.parent.rglob("*")
        if artifact.is_file()
    )


def test_initialized_submodule_is_captured_as_a_repository(tmp_path: Path):
    source = tmp_path / "rosa"
    external = source / "external"
    dependency = tmp_path / "detector-host"
    submodule_source = tmp_path / "projection-kernel"
    work_dir = tmp_path / "experiment"
    external.mkdir(parents=True)
    _initialize_repository(dependency)
    _initialize_repository(submodule_source)

    kernel = submodule_source / "kernel.py"
    kernel.write_text("VERSION = 1\n", encoding="utf-8")
    _git(submodule_source, "add", ".")
    _git(submodule_source, "commit", "--quiet", "-m", "kernel baseline")
    (dependency / "README.md").write_text("host\n", encoding="utf-8")
    _git(dependency, "add", ".")
    _git(dependency, "commit", "--quiet", "-m", "host baseline")
    _git(
        dependency,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "--quiet",
        os.fspath(submodule_source),
        "vendor/projection-kernel",
    )
    _git(dependency, "commit", "--quiet", "-am", "add submodule")
    checked_out_submodule = dependency / "vendor" / "projection-kernel"
    (checked_out_submodule / "kernel.py").write_text("VERSION = 2\n", encoding="utf-8")
    (external / "detector_host").symlink_to(dependency)

    reference = capture_external_provenance(source, work_dir)
    manifest_path = work_dir / reference["manifest"]
    manifest = json.loads(manifest_path.read_bytes())
    parent = next(
        repo for repo in manifest["repositories"] if repo["root"] == str(dependency)
    )
    submodule = parent["submodules"][0]
    child = next(
        repo
        for repo in manifest["repositories"]
        if repo["id"] == submodule["repository_id"]
    )

    assert submodule["path"] == "vendor/projection-kernel"
    assert submodule["resolved_path"] == os.fspath(checked_out_submodule.resolve())
    assert child["root"] == os.fspath(checked_out_submodule.resolve())
    assert (
        child["head"]
        == _git(checked_out_submodule, "rev-parse", "HEAD").decode().strip()
    )
    child_patch = (
        manifest_path.parent / child["artifacts"]["tracked.patch"]["path"]
    ).read_bytes()
    assert b"+VERSION = 2" in child_patch
