#!/usr/bin/env python3
"""
Thin shell — LMDB+BM25 implementation lives in hermes_state_lmdb.py.

Re-exports SessionDB, LMDB helpers, and BM25 index from hermes_state_lmdb.
The SQLite backward-compat helpers (apply_wal_with_fallback etc.) are kept
here because they are imported by kanban_db.py and other SQLite consumers.
"""

import logging
import sqlite3
import threading
from typing import Optional

from hermes_state_lmdb import (
    SessionDB,
    DEFAULT_DB_PATH,
    DEFAULT_BM25_PATH,
    SCHEMA_VERSION,
    _LMDB_SESSIONS,
    _LMDB_MESSAGES,
    _LMDB_MSG_INDEX,
    _LMDB_META,
    _lmdb_get_bytes,
    _lmdb_put,
    _lmdb_get_json,
    _lmdb_delete,
    _lmdb_iter_prefix,
    _lmdb_count_prefix,
    _encode_content,
    _decode_content,
    _MAX_TITLE_LENGTH,
    _BM25_MAX_TEXT_LENGTH,
    _sanitize_title,
    _contains_cjk,
    _count_cjk,
    _sanitize_fts5_query,
    _lmdb_env_registry,
    _lmdb_env_lock,
    _get_lmdb_env,
    _release_lmdb_env,
    _bm25_singleton,
    _bm25_singleton_lock,
    _get_bm25_index,
    __all__ as _lmdb_all,
)

logger = logging.getLogger(__name__)

# ── Backward-compat: SQLite WAL helpers kept for kanban_db ──
# kanban_db.py still uses SQLite and imports apply_wal_with_fallback.
_WAL_INCOMPAT_MARKERS = (
    "locking protocol",
    "not authorized",
    "disk i/o error",
)

_last_init_error: Optional[str] = None
_last_init_error_lock = threading.Lock()
_wal_fallback_warned_paths: set[str] = set()
_wal_fallback_warned_lock = threading.Lock()


def _set_last_init_error(msg: Optional[str]) -> None:
    global _last_init_error
    with _last_init_error_lock:
        _last_init_error = msg


def get_last_init_error() -> Optional[str]:
    return _last_init_error


def format_session_db_unavailable(prefix: str = "Session database not available") -> str:
    cause = get_last_init_error()
    if not cause:
        return f"{prefix}."
    return f"{prefix}: {cause}."


def _on_disk_journal_mode(conn: sqlite3.Connection) -> Optional[str]:
    """Read the journal mode from the SQLite DB header on disk.

    Returns the mode string (e.g. ``"wal"``, ``"delete"``), or ``None``
    if the value cannot be determined (new DB, or PRAGMA read failed).
    """
    try:
        row = conn.execute("PRAGMA journal_mode").fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    mode = row[0]
    if isinstance(mode, bytes):  # defensive: sqlite3 occasionally returns bytes
        try:
            mode = mode.decode("ascii")
        except UnicodeDecodeError:
            return None
    return str(mode).strip().lower() if mode is not None else None


def apply_wal_with_fallback(
    conn,
    *,
    db_label: str = "state.db",
) -> str:
    """Set journal_mode=WAL on a SQLite connection, falling back to DELETE.

    Kept for kanban_db.py which still uses SQLite.
    """
    # Read-only probe — no flock, no checkpoint, no WAL/SHM unlink.
    # Skipping the set-pragma prevents WAL-init from unlinking files other connections hold open.
    try:
        current_mode = conn.execute("PRAGMA journal_mode").fetchone()
        if current_mode and current_mode[0] == "wal":
            return "wal"
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute("PRAGMA journal_mode=WAL")
        return "wal"
    except Exception as exc:
        msg = str(exc).lower()
        if not any(marker in msg for marker in _WAL_INCOMPAT_MARKERS):
            raise
        with _wal_fallback_warned_lock:
            if db_label not in _wal_fallback_warned_paths:
                _wal_fallback_warned_paths.add(db_label)
                logger.warning(
                    "%s: WAL unsupported on this filesystem (%s), "
                    "falling back to journal_mode=DELETE",
                    db_label, exc,
                )
        conn.execute("PRAGMA journal_mode=DELETE")
        return "delete"


# ── __all__ ──
# Re-export everything from hermes_state_lmdb plus the SQLite helpers defined here.
__all__ = _lmdb_all + [
    "_set_last_init_error",
    "get_last_init_error",
    "format_session_db_unavailable",
    "_on_disk_journal_mode",
    "apply_wal_with_fallback",
    "_WAL_INCOMPAT_MARKERS",
    "_last_init_error",
    "_wal_fallback_warned_paths",
    "_wal_fallback_warned_lock",
]
