"""Deterministic, read-only provenance snapshots for external repositories.

The snapshot intentionally stores the contents of tracked changes only, as a
Git binary patch.  Untracked files are represented by metadata and SHA256
digests; they are never copied into an experiment directory.  Patch sections
that look like credentials or belong to sensitive paths are also replaced by
their digest so provenance capture cannot turn a work directory into a secret
archive.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any

_SCHEMA_VERSION = 1
_SECRET_LITERAL_PATTERNS = (
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\b(?:gh[opusr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(rb"(?i)\bBearer[ \t]+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(rb"(?i)\bhttps?://[^/\s:@]+:[^/\s@]+@"),
    re.compile(
        rb"(?im)^[+ -].{0,160}\b(?:password|passwd|api[_-]?key|"
        rb"client[_-]?secret|access[_-]?token|refresh[_-]?token)\b"
        rb"[ \t]*[:=][ \t]*[rubfRUBF]*['\"][^'\"\r\n]{6,}['\"]"
    ),
)
_ALWAYS_SENSITIVE_COMPONENTS = frozenset(
    {
        ".aws",
        ".gnupg",
        ".ssh",
        "credentials",
        "secrets",
    }
)
_TOP_LEVEL_DATA_COMPONENTS = frozenset(
    {"checkpoints", "data", "datasets", "outputs", "weights", "work_dirs"}
)
_SENSITIVE_SUFFIXES = frozenset(
    {
        ".ckpt",
        ".jks",
        ".key",
        ".kdbx",
        ".p12",
        ".pem",
        ".pfx",
        ".pkl",
        ".pt",
        ".pth",
    }
)


class _GitCommandError(RuntimeError):
    def __init__(self, operation: str, returncode: int, stderr: bytes) -> None:
        message = stderr.decode("utf-8", errors="replace").strip()
        super().__init__(
            f"git {operation} failed with exit code {returncode}: " f"{message[:500]}"
        )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    options = {
        "ensure_ascii": True,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _git(repository: Path, operation: str, *arguments: str) -> bytes:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    process = subprocess.run(
        [
            "git",
            "--no-pager",
            "-c",
            "core.quotePath=true",
            "-C",
            os.fspath(repository),
            *arguments,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    if process.returncode:
        raise _GitCommandError(operation, process.returncode, process.stderr)
    return process.stdout


def _safe_resolve(path: Path) -> str | None:
    try:
        return os.fspath(path.resolve(strict=False))
    except (OSError, RuntimeError):
        return None


def _entry_kind(path: Path) -> str:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return "missing"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "file"
    return "special"


def _find_git_root(path: Path) -> Path | None:
    if not path.exists() or not path.is_dir():
        return None
    try:
        output = _git(path, "rev-parse", "rev-parse", "--show-toplevel")
    except (OSError, _GitCommandError):
        return None
    return Path(os.fsdecode(output.rstrip(b"\n"))).resolve()


def _source_revision(source: Path) -> dict[str, Any]:
    """Identify tracked ROSA source state without inspecting untracked output."""

    git_root = _find_git_root(source)
    if git_root is None:
        return {
            "head": None,
            "branch": None,
            "tracked_dirty": None,
            "tracked_diff_sha256": None,
            "error": "source_root is not in a Git worktree",
        }

    last_revision: dict[str, Any] | None = None
    for attempt in range(1, 4):
        try:
            head = _git(
                git_root,
                "source rev-parse",
                "rev-parse",
                "--verify",
                "HEAD",
            ).strip()
            try:
                branch_output = _git(
                    git_root,
                    "source symbolic-ref",
                    "symbolic-ref",
                    "--quiet",
                    "--short",
                    "HEAD",
                )
                branch = os.fsdecode(branch_output.strip()) or None
            except _GitCommandError:
                branch = None
            tracked_diff = _git(
                git_root,
                "source diff",
                "diff",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                "--no-textconv",
                "HEAD",
                "--",
            )
            revision = {
                "head": os.fsdecode(head),
                "branch": branch,
                "tracked_dirty": bool(tracked_diff),
                "tracked_diff_sha256": _sha256(tracked_diff),
                "stable_capture": False,
                "capture_attempts": attempt,
            }
            last_revision = revision
            end_head = _git(
                git_root,
                "source rev-parse",
                "rev-parse",
                "--verify",
                "HEAD",
            ).strip()
            end_diff = _git(
                git_root,
                "source diff",
                "diff",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                "--no-textconv",
                "HEAD",
                "--",
            )
            revision["stable_capture"] = head == end_head and revision[
                "tracked_diff_sha256"
            ] == _sha256(end_diff)
            if revision["stable_capture"]:
                return revision
        except (OSError, _GitCommandError) as error:
            return {
                "head": None,
                "branch": None,
                "tracked_dirty": None,
                "tracked_diff_sha256": None,
                "error": f"{type(error).__name__}: {error}",
            }

    assert last_revision is not None
    return last_revision


def read_source_revision(source_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Return the stable, read-only source identity used by resume contracts."""

    return _source_revision(Path(source_root).resolve())


def _decode_paths(output: bytes, *, sort_paths: bool = True) -> list[str]:
    paths = [os.fsdecode(item) for item in output.split(b"\0") if item]
    return sorted(paths, key=os.fsencode) if sort_paths else paths


def _is_sensitive_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    parts = tuple(part.lower() for part in normalized.split("/") if part)
    if any(part in _ALWAYS_SENSITIVE_COMPONENTS for part in parts):
        return True
    if parts and parts[0] in _TOP_LEVEL_DATA_COMPONENTS:
        return True
    name = parts[-1] if parts else ""
    if name == ".env" or name.startswith(".env."):
        return True
    if Path(name).suffix.lower() in _SENSITIVE_SUFFIXES:
        return True
    return any(
        marker in name
        for marker in ("credential", "private_key", "private-key", "secret")
    )


def _contains_secret_literal(section: bytes) -> bool:
    return any(pattern.search(section) for pattern in _SECRET_LITERAL_PATTERNS)


def _split_patch(patch: bytes) -> list[bytes]:
    starts = [match.start() for match in re.finditer(rb"(?m)^diff --git ", patch)]
    if not starts:
        return [] if not patch else [patch]
    starts.append(len(patch))
    return [
        patch[starts[index] : starts[index + 1]] for index in range(len(starts) - 1)
    ]


def _sanitize_tracked_patch(
    patch: bytes,
    changed_paths: list[str],
) -> tuple[bytes, list[dict[str, Any]]]:
    sections = _split_patch(patch)
    if not sections:
        return patch, []

    # ``--name-only`` and patch sections share Git's diff order.  If an
    # unusual Git version violates that contract, use a stable synthetic name
    # and still apply content-level secret detection.
    safe_sections: list[bytes] = []
    omitted: list[dict[str, Any]] = []
    for index, section in enumerate(sections):
        path = (
            changed_paths[index]
            if index < len(changed_paths)
            else f"<unparsed-diff-section-{index}>"
        )
        header = section.split(b"\n", 1)[0].decode("utf-8", errors="replace")
        sensitive_path = _is_sensitive_path(path) or any(
            _is_sensitive_path(token.removeprefix("a/").removeprefix("b/"))
            for token in header.split()[2:4]
        )
        reason = None
        if sensitive_path:
            reason = "sensitive_path"
        elif _contains_secret_literal(section):
            reason = "credential_literal"
        if reason is None:
            safe_sections.append(section)
        else:
            omitted.append(
                {
                    "path": path,
                    "reason": reason,
                    "diff_section_sha256": _sha256(section),
                    "diff_section_size": len(section),
                }
            )
    return b"".join(safe_sections), omitted


def _untracked_manifest(repository: Path, paths: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative_path in paths:
        path = repository / relative_path
        kind = _entry_kind(path)
        record: dict[str, Any] = {
            "path": relative_path,
            "type": kind,
        }
        try:
            if kind == "file":
                digest, size = _sha256_file(path)
                record.update({"sha256": digest, "size": size})
            elif kind == "symlink":
                target = os.readlink(path)
                target_bytes = os.fsencode(target)
                record.update(
                    {
                        "link_target": target,
                        "resolved_path": _safe_resolve(path),
                        "sha256": _sha256(target_bytes),
                        "size": len(target_bytes),
                    }
                )
            elif kind == "missing":
                record["error"] = "file disappeared during capture"
            else:
                record["error"] = "special files are not read"
        except OSError as error:
            record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)
    return records


def _indexed_links(
    repository: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output = _git(repository, "ls-files", "ls-files", "-s", "-z")
    symlinks: list[dict[str, Any]] = []
    submodules: list[dict[str, Any]] = []
    for entry in output.split(b"\0"):
        if not entry:
            continue
        metadata, separator, raw_path = entry.partition(b"\t")
        if not separator:
            continue
        fields = metadata.split()
        if len(fields) != 3:
            continue
        mode, object_id, _stage = fields
        relative_path = os.fsdecode(raw_path)
        path = repository / relative_path
        if mode == b"120000":
            record: dict[str, Any] = {
                "path": relative_path,
                "resolved_path": _safe_resolve(path),
            }
            try:
                record["link_target"] = os.readlink(path)
            except OSError:
                record["link_target"] = None
            symlinks.append(record)
        elif mode == b"160000":
            submodules.append(
                {
                    "path": relative_path,
                    "index_commit": os.fsdecode(object_id),
                    "resolved_path": _safe_resolve(path),
                    "initialized": (path / ".git").exists(),
                }
            )
    symlinks.sort(key=lambda item: os.fsencode(item["path"]))
    submodules.sort(key=lambda item: os.fsencode(item["path"]))
    return symlinks, submodules


def _artifact_record(relative_path: str, data: bytes) -> dict[str, Any]:
    return {
        "path": relative_path,
        "sha256": _sha256(data),
        "size": len(data),
    }


def _capture_repository(
    repository: Path,
    repository_id: str,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    artifact_root = f"repositories/{repository_id}"
    last_capture: tuple[dict[str, Any], dict[str, bytes]] | None = None

    for attempt in range(1, 4):
        head = _git(repository, "rev-parse", "rev-parse", "--verify", "HEAD").strip()
        try:
            branch_output = _git(
                repository,
                "symbolic-ref",
                "symbolic-ref",
                "--quiet",
                "--short",
                "HEAD",
            )
            branch = os.fsdecode(branch_output.strip()) or None
        except _GitCommandError:
            branch = None

        status = _git(
            repository,
            "status",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
        )
        raw_patch = _git(
            repository,
            "diff",
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD",
            "--",
        )
        changed_paths = _decode_paths(
            _git(
                repository,
                "diff --name-only",
                "diff",
                "--name-only",
                "-z",
                "HEAD",
                "--",
            ),
            sort_paths=False,
        )
        tracked_patch, omitted = _sanitize_tracked_patch(raw_patch, changed_paths)
        untracked_paths = _decode_paths(
            _git(
                repository,
                "ls-files --others",
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            )
        )
        untracked = _untracked_manifest(repository, untracked_paths)
        tracked_symlinks, submodules = _indexed_links(repository)
        try:
            submodule_status = _git(
                repository,
                "submodule status",
                "submodule",
                "status",
                "--recursive",
            )
        except _GitCommandError as error:
            submodule_status = f"ERROR: {error}\n".encode("utf-8")

        untracked_bytes = _json_bytes(untracked)
        artifacts = {
            f"{artifact_root}/status.porcelain-v1.z": status,
            f"{artifact_root}/tracked.patch": tracked_patch,
            f"{artifact_root}/untracked.json": untracked_bytes,
            f"{artifact_root}/submodules.status": submodule_status,
        }
        artifact_metadata = {
            name.rsplit("/", 1)[-1]: _artifact_record(name, data)
            for name, data in artifacts.items()
        }
        record: dict[str, Any] = {
            "id": repository_id,
            "root": os.fspath(repository),
            "head": os.fsdecode(head),
            "branch": branch,
            "dirty": bool(status),
            "status_entry_count": len([item for item in status.split(b"\0") if item]),
            "tracked_changed_paths": changed_paths,
            "omitted_tracked_diff_sections": omitted,
            "untracked_file_count": len(untracked),
            "tracked_symlinks": tracked_symlinks,
            "submodules": submodules,
            "artifacts": artifact_metadata,
            "capture_attempts": attempt,
        }
        last_capture = record, artifacts

        # A second diff and status check catches ordinary writes racing with
        # capture without creating a lock in the user's repository.
        end_head = _git(
            repository, "rev-parse", "rev-parse", "--verify", "HEAD"
        ).strip()
        end_status = _git(
            repository,
            "status",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
        )
        end_patch = _git(
            repository,
            "diff",
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD",
            "--",
        )
        stable = (
            head == end_head
            and status == end_status
            and _sha256(raw_patch) == _sha256(end_patch)
        )
        record["stable_capture"] = stable
        if stable:
            return record, artifacts

    assert last_capture is not None
    return last_capture


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return (slug or "repository")[:100]


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def capture_external_provenance(
    source_root: str | os.PathLike[str],
    work_dir: str | os.PathLike[str],
    *,
    external_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Capture read-only external dependency state in ``work_dir``.

    Returns the small, path-relative reference suitable for checkpoint
    metadata.  The same external state always produces the same snapshot ID;
    rerunning capture is idempotent.
    """

    source = Path(source_root).resolve()
    work = Path(work_dir).resolve()
    external = (
        Path(external_dir).resolve()
        if external_dir is not None
        else source / "external"
    )
    entries: list[dict[str, Any]] = []
    repositories: list[dict[str, Any]] = []
    artifacts: dict[str, bytes] = {}
    repository_ids: dict[str, str] = {}
    source_revision = _source_revision(source)

    def add_repository(path: Path, suggested_id: str) -> str | None:
        git_root = _find_git_root(path)
        if git_root is None:
            return None
        # A plain vendored directory under ``external/`` inherits the ROSA
        # worktree.  Capturing the whole parent repository here would be both
        # redundant and much broader than the requested external dependency.
        if git_root == source and path.resolve() != source:
            return None
        key = os.fspath(git_root)
        if key in repository_ids:
            return repository_ids[key]
        repository_id = _slug(suggested_id)
        used_ids = {record["id"] for record in repositories}
        if repository_id in used_ids:
            repository_id = f"{repository_id}-{len(repositories):02d}"
        repository_ids[key] = repository_id
        try:
            record, repo_artifacts = _capture_repository(git_root, repository_id)
        except (OSError, _GitCommandError) as error:
            record = {
                "id": repository_id,
                "root": key,
                "capture_error": f"{type(error).__name__}: {error}",
            }
            repo_artifacts = {}
        repositories.append(record)
        artifacts.update(repo_artifacts)

        for submodule in record.get("submodules", []):
            submodule_path = git_root / submodule["path"]
            child_id = add_repository(
                submodule_path,
                f"{repository_id}--submodule--{submodule['path']}",
            )
            submodule["repository_id"] = child_id
        return repository_id

    if external.exists() and external.is_dir():
        external_entries = sorted(
            external.iterdir(), key=lambda path: os.fsencode(path.name)
        )
    else:
        external_entries = []

    for index, path in enumerate(external_entries):
        kind = _entry_kind(path)
        try:
            entry_path = os.fspath(path.relative_to(source))
        except ValueError:
            entry_path = os.fspath(path)
        entry: dict[str, Any] = {
            "name": path.name,
            "entry_path": entry_path,
            "type": kind,
            "resolved_path": _safe_resolve(path),
            "target_exists": path.exists(),
        }
        if kind == "symlink":
            try:
                entry["link_target"] = os.readlink(path)
            except OSError:
                entry["link_target"] = None
        resolved = Path(entry["resolved_path"]) if entry["resolved_path"] else path
        entry["repository_id"] = add_repository(
            resolved,
            f"{index:02d}-{path.name}",
        )
        entries.append(entry)

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "source_root": os.fspath(source),
        "source_revision": source_revision,
        "external_root": os.fspath(external),
        "external_entries": entries,
        "repositories": repositories,
        "privacy": {
            "tracked_patch": "binary Git diff with sensitive sections omitted",
            "untracked_files": "metadata and SHA256 only; contents are not copied",
            "ignored_files": "not enumerated",
            "environment_and_git_remotes": "not captured",
        },
    }
    snapshot_sha256 = _sha256(_json_bytes(payload, pretty=False))
    manifest = {**payload, "snapshot_sha256": snapshot_sha256}
    manifest_bytes = _json_bytes(manifest)
    snapshot_relative = Path("provenance") / "external" / "snapshots" / snapshot_sha256
    snapshot_root = work / snapshot_relative

    for relative_path, data in sorted(artifacts.items()):
        _atomic_write(snapshot_root / relative_path, data)
    manifest_path = snapshot_root / "manifest.json"
    _atomic_write(manifest_path, manifest_bytes)

    result = {
        "schema_version": _SCHEMA_VERSION,
        "snapshot_sha256": snapshot_sha256,
        "manifest": os.fspath(snapshot_relative / "manifest.json"),
        "manifest_sha256": _sha256(manifest_bytes),
        "source_revision": source_revision,
    }
    _atomic_write(
        work / "provenance" / "external" / "latest.json",
        _json_bytes(result),
    )
    return result


__all__ = ["capture_external_provenance", "read_source_revision"]
