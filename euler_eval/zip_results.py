"""Write result JSON into zip archives, once per archive.

Adding an entry to a zip rewrites its central directory; replacing one, as a
re-run does, rewrites the whole archive. A split sweep or a set of experiments
saves many results into the same prediction archive, so inside
:func:`batched_zip_writes` they are spooled to local temporary files and each
archive is written once, when the block ends.

Each write holds :func:`zip_lock` where the filesystem supports one, so jobs
writing into the same archive take turns instead of corrupting it.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import shutil
import sys
import tempfile
import zipfile
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator, Mapping

from ds_crawler.zip_utils import get_zip_root_prefix

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None

#: errno values of a lock another process holds.
_LOCK_HELD = {errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK}


def json_bytes(data: Any) -> bytes:
    """Serialize a result the way it is stored."""
    return json.dumps(data, indent=2).encode("utf-8")


def zip_entry_name(
    zip_path: Path, internal_name: str, prefix: str | None = None
) -> str:
    """Place *internal_name* inside a zip's wrapper folder, if it has one.

    ds-crawler treats a single top-level folder named after the archive
    (``model.zip`` -> ``model/``) as the dataset root. Writing beside that
    folder would end the single-root layout and hide ``.ds_crawler/``.

    Args:
        prefix: The archive's root prefix, when already read.
    """
    if prefix is None:
        prefix = get_zip_root_prefix(zip_path)
    if (
        prefix
        and prefix.rstrip("/") == zip_path.stem
        and not internal_name.startswith(prefix)
    ):
        return prefix + internal_name
    return internal_name


def write_zip_entries(zip_path: Path, entries: Mapping[str, bytes | Path]) -> None:
    """Write *entries* into an existing zip archive in one pass.

    Args:
        zip_path: The archive.
        entries: Entry name to its content, or to a file holding it. Names
            are used as given (see :func:`zip_entry_name`).

    New names are appended, which rewrites only the central directory. When
    any name exists, the archive is rebuilt once, without the replaced
    entries, in a temporary file beside it that then replaces it, keeping
    its permissions.
    """
    if not entries:
        return

    def add(archive: zipfile.ZipFile) -> None:
        for name, content in entries.items():
            data = content.read_bytes() if isinstance(content, Path) else content
            archive.writestr(name, data)

    target = Path(zip_path).resolve()
    with zipfile.ZipFile(target, "r") as archive:
        replaced = not set(entries).isdisjoint(archive.namelist())
    if not replaced:
        with zipfile.ZipFile(target, "a", zipfile.ZIP_DEFLATED) as archive:
            add(archive)
        return

    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with zipfile.ZipFile(target, "r") as src, zipfile.ZipFile(
            tmp, "w", zipfile.ZIP_DEFLATED
        ) as dst:
            for item in src.infolist():
                if item.filename not in entries:
                    dst.writestr(item, src.read(item.filename))
            add(dst)
        shutil.copymode(target, tmp)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def lock_path(zip_path: Path) -> Path:
    """Return the hidden lock file :func:`zip_lock` uses beside *zip_path*."""
    return zip_path.with_name(f".{zip_path.name}.lock")


def _write_unlocked(zip_path: Path, reason: object) -> None:
    print(
        f"Note: writing into {zip_path} without a lock ({reason}); another "
        "process writing into it at the same time can corrupt it.",
        file=sys.stderr,
    )


@contextlib.contextmanager
def zip_lock(zip_path: Path) -> Iterator[bool]:
    """Hold an exclusive lock for writing into *zip_path*, where possible.

    The lock is ``flock`` on a hidden ``.<name>.lock`` file beside the
    archive, which is left in place. A process that finds it held prints that
    it is waiting and blocks until the holder's write is done; a holder that
    dies releases it. Where no lock can be had (no ``fcntl``, a filesystem
    without ``flock``, a lock file that cannot be created), a note is printed
    and the block runs unlocked.

    Yields:
        Whether the archive is locked.
    """
    zip_path = Path(zip_path)
    if fcntl is None:
        _write_unlocked(zip_path, "no fcntl on this platform")
        yield False
        return
    lock = lock_path(zip_path)
    try:
        fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o666)
    except PermissionError:
        # Another user's lock file: local filesystems lock read-only files.
        try:
            fd = os.open(lock, os.O_RDONLY)
        except OSError as e:
            _write_unlocked(zip_path, e)
            yield False
            return
    except OSError as e:
        _write_unlocked(zip_path, e)
        yield False
        return
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno not in _LOCK_HELD:
                _write_unlocked(zip_path, e)
                yield False
                return
            print(
                f"Waiting for another process writing into {zip_path}",
                flush=True,
            )
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield True
    finally:
        os.close(fd)


def _write_into(zip_path: Path, entries: Mapping[str, bytes | Path]) -> None:
    """Write *entries* inside the archive's wrapper folder, under its lock."""
    target = Path(zip_path).resolve()
    with zip_lock(target):
        prefix = get_zip_root_prefix(target)
        write_zip_entries(
            target,
            {
                zip_entry_name(target, name, prefix): content
                for name, content in entries.items()
            },
        )


class ZipWriteError(OSError):
    """Results could not be written into one or more zip archives."""


class _ZipWriteBatch:
    """Results bound for zip archives, spooled to local temporary files."""

    def __init__(self) -> None:
        self._spool: Path | None = None
        self._pending: dict[Path, dict[str, Path]] = {}
        self._count = 0

    def add(self, zip_path: Path, internal_name: str, data: Any) -> None:
        if self._spool is None:
            self._spool = Path(tempfile.mkdtemp(prefix="euler-eval-results-"))
        self._count += 1
        spooled = self._spool / f"{self._count}.json"
        spooled.write_bytes(json_bytes(data))
        # A later result for the same entry replaces the earlier one.
        self._pending.setdefault(Path(zip_path), {})[internal_name] = spooled

    def write(self) -> None:
        """Write each archive's results; keep the spool if any archive fails.

        Raises:
            ZipWriteError: Naming the archives that could not be written and
                where their results were kept.
        """
        failed = []
        for zip_path, entries in self._pending.items():
            print(
                f"\nWriting {len(entries)} result file"
                f"{'' if len(entries) == 1 else 's'} into {zip_path}"
            )
            try:
                _write_into(zip_path, entries)
            except Exception as e:
                failed.append(f"{zip_path} ({e})")
        if failed:
            raise ZipWriteError(
                f"Could not write results into {'; '.join(failed)}. They are "
                f"kept in {self._spool}, numbered in the order they were saved."
            )
        if self._spool is not None:
            shutil.rmtree(self._spool, ignore_errors=True)


_BATCH: ContextVar[_ZipWriteBatch | None] = ContextVar("zip_write_batch", default=None)


@contextlib.contextmanager
def batched_zip_writes() -> Iterator[None]:
    """Hold results bound for zip archives until the block ends.

    Each archive is then written once (see :func:`write_zip_entries`).
    Results saved before an exception are written too; when that write
    fails as well, its error is printed and the block's exception raised.

    Raises:
        ZipWriteError: When the block succeeded but an archive could not be
            written.
    """
    batch = _ZipWriteBatch()
    token = _BATCH.set(batch)
    try:
        yield
    except BaseException:
        _BATCH.reset(token)
        try:
            batch.write()
        except ZipWriteError as e:
            print(f"Error: {e}", file=sys.stderr)
        raise
    _BATCH.reset(token)
    batch.write()


def zip_writes_deferred() -> bool:
    """Return whether results saved into a zip now wait for the batch end."""
    return _BATCH.get() is not None


def save_json_to_zip(zip_path: Path, internal_name: str, data: Any) -> None:
    """Save *data* as *internal_name* inside *zip_path*'s wrapper folder.

    Inside :func:`batched_zip_writes` the write waits for the block's end.
    Either way it holds :func:`zip_lock`.
    """
    batch = _BATCH.get()
    if batch is not None:
        batch.add(zip_path, internal_name, data)
    else:
        _write_into(zip_path, {internal_name: json_bytes(data)})
