#!/usr/bin/env python3
"""
LMDB + BM25 State Store for Hermes Agent.

Replaces the original SQLite + FTS5 implementation. Uses LMDB for KV storage
(抗并发 + 内存映射) and BM25 for full-text search.

Key design decisions:
- LMDB named databases for sessions, messages, meta
- BM25 index for full-text search (saved as gzipped JSON alongside LMDB)
- Same public API as the SQLite version — no consumer code changes needed.
- Thread-safe via threading.Lock (coordinating LMDB transactions + BM25 updates).
"""

import json
import logging
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import lmdb

from hermes_bm25 import BM25Index, _fts5_to_bm25_query, contains_cjk, tokenize
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ── Path ────────────────────────────────────────────────
# state.lmdb replaces state.db
DEFAULT_DB_PATH = get_hermes_home() / "state.lmdb"
DEFAULT_BM25_PATH = DEFAULT_DB_PATH.with_suffix(".lmdb.bm25.gz")

SCHEMA_VERSION = 14  # bumped from 13 (SQLite version) for the LMDB migration

# ── LMDB named databases ───────────────────────────────
_LMDB_SESSIONS = "sessions"
_LMDB_MESSAGES = "messages"
_LMDB_MSG_INDEX = "msg_idx"  # msg_id → session_id
_LMDB_META = "meta"

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


def apply_wal_with_fallback(
    conn,
    *,
    db_label: str = "state.db",
) -> str:
    """Set journal_mode=WAL on a SQLite connection, falling back to DELETE.

    Kept for kanban_db.py which still uses SQLite.
    """
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


# ── LMDB helpers ────────────────────────────────────────

def _lmdb_get_bytes(txn: lmdb.Transaction, db: Any, key: str) -> Optional[bytes]:
    """Safely get bytes from LMDB within a transaction."""
    return txn.get(key.encode("utf-8"), db=db)


def _lmdb_put(txn: lmdb.Transaction, db: Any, key: str, value: Any) -> None:
    """JSON-serialize and store a value in LMDB."""
    encoded = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
    txn.put(key.encode("utf-8"), encoded, db=db)


def _lmdb_get_json(txn: lmdb.Transaction, db: Any, key: str) -> Optional[Any]:
    """Read and JSON-deserialize a value from LMDB."""
    raw = txn.get(key.encode("utf-8"), db=db)
    if raw is None:
        return None
    return json.loads(raw.decode("utf-8"))


def _lmdb_delete(txn: lmdb.Transaction, db: Any, key: str) -> None:
    txn.delete(key.encode("utf-8"), db=db)


def _lmdb_iter_prefix(txn: lmdb.Transaction, db: Any, prefix: str) -> List[Tuple[str, Any]]:
    """Iterate LMDB entries with a key prefix, returning [(key, value), ...]."""
    results: List[Tuple[str, Any]] = []
    cursor = txn.cursor(db=db)
    prefix_bytes = prefix.encode("utf-8")
    if not cursor.set_range(prefix_bytes):
        return results
    for key_bytes, value_bytes in cursor:
        if not key_bytes.startswith(prefix_bytes):
            break
        try:
            key_str = key_bytes.decode("utf-8")
            value = json.loads(value_bytes.decode("utf-8"))
            results.append((key_str, value))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    cursor.close()
    return results


def _lmdb_count_prefix(txn: lmdb.Transaction, db: Any, prefix: str) -> int:
    """Count LMDB entries with a key prefix."""
    count = 0
    cursor = txn.cursor(db=db)
    prefix_bytes = prefix.encode("utf-8")
    if not cursor.set_range(prefix_bytes):
        return 0
    for key_bytes, _ in cursor:
        if not key_bytes.startswith(prefix_bytes):
            break
        count += 1
    cursor.close()
    return count


# ── Content encoding / decoding ─────────────────────────

def _encode_content(content: Any) -> Optional[str]:
    """Encode message content for storage.

    Multimodal content (list of dicts) is JSON-encoded.
    Strings are stored as-is (None stays None).
    """
    if content is None:
        return None
    if isinstance(content, (list, dict)):
        return json.dumps(content, ensure_ascii=False, default=str)
    if isinstance(content, str):
        return content
    return str(content)


def _decode_content(stored: Any) -> Any:
    """Decode message content from storage.

    JSON-encoded lists/dicts are decoded; plain strings are returned as-is.
    Never returns None — if stored is None, returns "" to prevent
    "content should be a string or a list" 400 errors from strict
    OpenAI-compatible providers (DeepSeek, etc.).
    """
    if stored is None:
        return ""
    if isinstance(stored, str) and stored.startswith("["):
        try:
            return json.loads(stored)
        except (json.JSONDecodeError, ValueError):
            pass
    if isinstance(stored, str) and stored.startswith("{"):
        try:
            return json.loads(stored)
        except (json.JSONDecodeError, ValueError):
            pass
    return stored


# ── Sanitize helpers ────────────────────────────────────

_MAX_TITLE_LENGTH = 100
# BM25 indexing limit — truncate very long messages to this many chars.
# Full-text search quality is preserved with the first ~20K chars.
_BM25_MAX_TEXT_LENGTH = 20000

def _sanitize_title(title: Optional[str]) -> Optional[str]:
    """Validate and sanitize a session title."""
    if not title:
        return None
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', title)
    cleaned = re.sub(
        r'[\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff\ufffc\ufff9-\ufffb]',
        '', cleaned,
    )
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if not cleaned:
        return None
    if len(cleaned) > _MAX_TITLE_LENGTH:
        raise ValueError(
            f"Title too long ({len(cleaned)} chars, max {_MAX_TITLE_LENGTH})"
        )
    return cleaned


# ── CJK helpers for search ─────────────────────────────

def _contains_cjk(s: str) -> bool:
    return contains_cjk(s)


def _count_cjk(s: str) -> int:
    return sum(1 for ch in s if ord(ch) > 0x2E80 and ord(ch) < 0x30000
               and any(0x4E00 <= ord(ch) <= 0x9FFF for _ in [1]))


def _sanitize_fts5_query(query: str) -> str:
    """Sanitize FTS5 query (adapted for BM25 compatibility).

    Removes characters that are valid in FTS5 but meaningless/noxious in BM25.
    """
    if not query:
        return ""
    # Remove leading/trailing whitespace
    query = query.strip()
    # BM25 doesn't need escaping — keep the raw text
    return query


# ── LMDB Environment Singleton ──────────────────────────
# Prevents "already open in this process" when multiple
# SessionDB instances (TUI, gateway, CLI) share the same LMDB.
_lmdb_env_registry: Dict[str, Tuple[lmdb.Environment, int]] = {}
_lmdb_env_lock = threading.Lock()
# BM25 singleton (shared across instances, lazy-loaded)
_bm25_singleton: Optional['BM25Index'] = None
_bm25_singleton_lock = threading.Lock()


def _get_lmdb_env(path: str, map_size: int = 2 * 1024 ** 3, max_dbs: int = 32) -> lmdb.Environment:
    with _lmdb_env_lock:
        if path in _lmdb_env_registry:
            env, ref = _lmdb_env_registry[path]
            _lmdb_env_registry[path] = (env, ref + 1)
            return env
        env = lmdb.open(path, map_size=map_size, max_dbs=max_dbs)
        _lmdb_env_registry[path] = (env, 1)
        return env


def _release_lmdb_env(path: str) -> None:
    with _lmdb_env_lock:
        if path not in _lmdb_env_registry:
            return
        env, ref = _lmdb_env_registry[path]
        if ref <= 1:
            env.close()
            del _lmdb_env_registry[path]
        else:
            _lmdb_env_registry[path] = (env, ref - 1)


def _get_bm25_index(path: str) -> Optional['BM25Index']:
    """Get the shared BM25 index (lazy-loaded singleton, safe against OOM).

    Returns None if the BM25 file is missing, too large to load, or corrupt.
    Session CRUD works without BM25; only full-text search is degraded.
    """
    global _bm25_singleton
    if _bm25_singleton is not None:
        return _bm25_singleton
    with _bm25_singleton_lock:
        if _bm25_singleton is not None:
            return _bm25_singleton
        try:
            _bm25_singleton = BM25Index.load(path)
        except Exception as exc:
            logger.warning("BM25 index load skipped (%s) — search degraded, CRUD unaffected", exc)
            _bm25_singleton = None
    return _bm25_singleton


# ── Session DB ──────────────────────────────────────────

class SessionDB:
    """
    LMDB-backed session storage with BM25 full-text search.

    Thread-safe: all public methods acquire self._lock before accessing LMDB/BM25.
    Multi-process safe: LMDB supports concurrent readers natively.
    """

    MAX_TITLE_LENGTH = _MAX_TITLE_LENGTH

    def __init__(self, db_path: Any = None):
        if db_path is not None:
            db_path = Path(db_path)
        self.db_path = db_path or DEFAULT_DB_PATH
        self.bm25_path = self.db_path.with_suffix(".lmdb.bm25.gz")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._env: Optional[lmdb.Environment] = None
        self._bm25: Optional[BM25Index] = None
        self._closed = False

        try:
            # Open LMDB environment (singleton — shares across instances)
            self._env = _get_lmdb_env(
                str(self.db_path),
                map_size=2 * 1024 * 1024 * 1024,  # 2GB map
                max_dbs=32,
            )

            # Create named databases
            self._sessions_db = self._env.open_db(key=_LMDB_SESSIONS.encode(), create=True)
            self._messages_db = self._env.open_db(key=_LMDB_MESSAGES.encode(), create=True)
            self._msg_idx_db = self._env.open_db(key=_LMDB_MSG_INDEX.encode(), create=True)
            self._meta_db = self._env.open_db(key=_LMDB_META.encode(), create=True)

            # Initialize schema version and counters
            with self._env.begin(write=True) as txn:
                ver_raw = _lmdb_get_bytes(txn, self._meta_db, "schema_version")
                if ver_raw is None:
                    _lmdb_put(txn, self._meta_db, "schema_version", SCHEMA_VERSION)
                    _lmdb_put(txn, self._meta_db, "next_msg_id", 1)

            # Open BM25 index (shared singleton — lazy-loaded on first use)
            self._bm25 = _get_bm25_index(str(self.bm25_path))
            bm25_docs = self._bm25.doc_count() if self._bm25 else 0
            logger.info(
                "SessionDB opened at %s (LMDB + BM25, %d docs indexed)",
                self.db_path, bm25_docs,
            )

        except Exception as exc:
            _set_last_init_error(f"{type(exc).__name__}: {exc}")
            raise

    # ── Internal helpers ──

    def _next_msg_id(self, txn: lmdb.Transaction) -> int:
        """Atomically increment and return the next message ID."""
        raw = _lmdb_get_bytes(txn, self._meta_db, "next_msg_id")
        current = int(raw.decode()) if raw else 1
        _lmdb_put(txn, self._meta_db, "next_msg_id", current + 1)
        return current

    def _msg_key(self, session_id: str, msg_id: int) -> str:
        return f"{session_id}:{msg_id}"

    def _parse_msg_key(self, key: str) -> Optional[Tuple[str, int]]:
        """Parse '<session_id>:<msg_id>' from a message key.
        Returns (session_id, msg_id) or None."""
        idx = key.rfind(":")
        if idx == -1:
            return None
        sid = key[:idx]
        try:
            mid = int(key[idx + 1:])
        except ValueError:
            return None
        return (sid, mid)

    def _session_exists(self, txn: lmdb.Transaction, session_id: str) -> bool:
        return txn.get(session_id.encode("utf-8"), db=self._sessions_db) is not None

    def _write_bm25_async(self) -> None:
        """Save BM25 index in a background thread to avoid blocking writes.

        Only persists when dirty. Safe to call after every write.
        """
        import threading as _thr
        bm25 = self._bm25
        path = str(self.bm25_path)
        if bm25 and bm25.is_dirty:
            t = _thr.Thread(target=bm25.save, args=(path,), daemon=True)
            t.start()

    # ── Session lifecycle ──

    def _insert_session_row(
        self,
        session_id: str,
        source: str,
        model: str = None,
        model_config: Dict[str, Any] = None,
        system_prompt: str = None,
        user_id: str = None,
        parent_session_id: str = None,
    ) -> None:
        """INSERT OR IGNORE equivalent for LMDB."""
        with self._lock:
            with self._env.begin(write=True) as txn:
                sdb = self._sessions_db
                existing = txn.get(session_id.encode("utf-8"), db=sdb)
                if existing is not None:
                    return  # Already exists (INSERT OR IGNORE)
                now = time.time()
                session_data = {
                    "id": session_id,
                    "source": source,
                    "user_id": user_id,
                    "model": model,
                    "model_config": json.dumps(model_config) if model_config else None,
                    "system_prompt": system_prompt,
                    "parent_session_id": parent_session_id,
                    "started_at": now,
                    "ended_at": None,
                    "end_reason": None,
                    "message_count": 0,
                    "tool_call_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "reasoning_tokens": 0,
                    "billing_provider": None,
                    "billing_base_url": None,
                    "billing_mode": None,
                    "estimated_cost_usd": None,
                    "actual_cost_usd": None,
                    "cost_status": None,
                    "cost_source": None,
                    "pricing_version": None,
                    "title": None,
                    "api_call_count": 0,
                    "handoff_state": None,
                    "handoff_platform": None,
                    "handoff_error": None,
                }
                _lmdb_put(txn, sdb, session_id, session_data)

    def create_session(self, session_id: str, source: str, **kwargs) -> str:
        """Create a new session record. Returns the session_id."""
        self._insert_session_row(session_id, source, **kwargs)
        return session_id

    def ensure_session(
        self,
        session_id: str,
        source: str = "unknown",
        model: str = None,
        **kwargs,
    ) -> str:
        """Ensure a session row exists (INSERT OR IGNORE)."""
        self._insert_session_row(session_id, source, model=model, **kwargs)
        return session_id

    def end_session(self, session_id: str, end_reason: str) -> None:
        """Mark a session as ended. First end_reason wins."""
        with self._lock:
            with self._env.begin(write=True) as txn:
                sdb = self._sessions_db
                session = _lmdb_get_json(txn, sdb, session_id)
                if session is None:
                    return
                if session.get("ended_at") is not None:
                    return  # Already ended
                session["ended_at"] = time.time()
                session["end_reason"] = end_reason
                _lmdb_put(txn, sdb, session_id, session)

    def reopen_session(self, session_id: str) -> None:
        """Clear ended_at/end_reason so a session can be resumed."""
        with self._lock:
            with self._env.begin(write=True) as txn:
                sdb = self._sessions_db
                session = _lmdb_get_json(txn, sdb, session_id)
                if session is None:
                    return
                session["ended_at"] = None
                session["end_reason"] = None
                _lmdb_put(txn, sdb, session_id, session)

    def update_system_prompt(self, session_id: str, system_prompt: str) -> None:
        """Store the full assembled system prompt snapshot."""
        with self._lock:
            with self._env.begin(write=True) as txn:
                sdb = self._sessions_db
                session = _lmdb_get_json(txn, sdb, session_id)
                if session is None:
                    return
                session["system_prompt"] = system_prompt
                _lmdb_put(txn, sdb, session_id, session)

    def update_token_counts(
        self,
        session_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
        actual_cost_usd: Optional[float] = None,
        cost_status: Optional[str] = None,
        cost_source: Optional[str] = None,
        pricing_version: Optional[str] = None,
        billing_provider: Optional[str] = None,
        billing_base_url: Optional[str] = None,
        billing_mode: Optional[str] = None,
        api_call_count: int = 0,
        absolute: bool = False,
    ) -> None:
        """Update token counters. absolute=True → set; False → increment."""
        with self._lock:
            with self._env.begin(write=True) as txn:
                sdb = self._sessions_db
                session = _lmdb_get_json(txn, sdb, session_id)
                if session is None:
                    # Auto-create (same behavior as SQLite version)
                    session = {
                        "id": session_id,
                        "source": "unknown",
                        "user_id": None,
                        "model": model,
                        "model_config": None,
                        "system_prompt": None,
                        "parent_session_id": None,
                        "started_at": time.time(),
                        "ended_at": None,
                        "end_reason": None,
                        "message_count": 0,
                        "tool_call_count": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_read_tokens": 0,
                        "cache_write_tokens": 0,
                        "reasoning_tokens": 0,
                        "billing_provider": None,
                        "billing_base_url": None,
                        "billing_mode": None,
                        "estimated_cost_usd": None,
                        "actual_cost_usd": None,
                        "cost_status": None,
                        "cost_source": None,
                        "pricing_version": None,
                        "title": None,
                        "api_call_count": 0,
                        "handoff_state": None,
                        "handoff_platform": None,
                        "handoff_error": None,
                    }
                if absolute:
                    session["input_tokens"] = input_tokens
                    session["output_tokens"] = output_tokens
                    session["cache_read_tokens"] = cache_read_tokens
                    session["cache_write_tokens"] = cache_write_tokens
                    session["reasoning_tokens"] = reasoning_tokens
                    session["estimated_cost_usd"] = estimated_cost_usd or 0
                    if actual_cost_usd is not None:
                        session["actual_cost_usd"] = actual_cost_usd
                    if cost_status is not None:
                        session["cost_status"] = cost_status
                    if cost_source is not None:
                        session["cost_source"] = cost_source
                    if pricing_version is not None:
                        session["pricing_version"] = pricing_version
                    if billing_provider is not None:
                        session["billing_provider"] = billing_provider
                    if billing_base_url is not None:
                        session["billing_base_url"] = billing_base_url
                    if billing_mode is not None:
                        session["billing_mode"] = billing_mode
                    if model is not None:
                        session["model"] = model if not session.get("model") else session["model"]
                    session["api_call_count"] = api_call_count
                else:
                    session["input_tokens"] = (session.get("input_tokens") or 0) + input_tokens
                    session["output_tokens"] = (session.get("output_tokens") or 0) + output_tokens
                    session["cache_read_tokens"] = (session.get("cache_read_tokens") or 0) + cache_read_tokens
                    session["cache_write_tokens"] = (session.get("cache_write_tokens") or 0) + cache_write_tokens
                    session["reasoning_tokens"] = (session.get("reasoning_tokens") or 0) + reasoning_tokens
                    session["estimated_cost_usd"] = (session.get("estimated_cost_usd") or 0) + (estimated_cost_usd or 0)
                    if actual_cost_usd is not None:
                        session["actual_cost_usd"] = (session.get("actual_cost_usd") or 0) + actual_cost_usd
                    if cost_status is not None:
                        session["cost_status"] = cost_status
                    if cost_source is not None:
                        session["cost_source"] = cost_source
                    if pricing_version is not None:
                        session["pricing_version"] = pricing_version
                    if billing_provider is not None:
                        session["billing_provider"] = billing_provider
                    if billing_base_url is not None:
                        session["billing_base_url"] = billing_base_url
                    if billing_mode is not None:
                        session["billing_mode"] = billing_mode
                    if model is not None:
                        session["model"] = model if not session.get("model") else session["model"]
                    session["api_call_count"] = (session.get("api_call_count") or 0) + api_call_count
                _lmdb_put(txn, sdb, session_id, session)

    # ── Session queries ──

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        if not session_id:
            return None
        with self._lock:
            with self._env.begin() as txn:
                return _lmdb_get_json(txn, self._sessions_db, session_id)

    def resolve_session_id(self, session_id_or_prefix: str) -> Optional[str]:
        """Resolve an exact or uniquely prefixed session ID."""
        if not session_id_or_prefix:
            return None
        with self._lock:
            with self._env.begin() as txn:
                sdb = self._sessions_db
                # Exact match
                exact = _lmdb_get_json(txn, sdb, session_id_or_prefix)
                if exact is not None:
                    return exact["id"]

                # Prefix search (sorted LMDB keys)
                prefix = session_id_or_prefix.encode("utf-8")
                matches = []
                cursor = txn.cursor(db=sdb)
                if cursor.set_range(prefix):
                    for key_bytes, _ in cursor:
                        if not key_bytes.startswith(prefix):
                            break
                        matches.append(key_bytes.decode("utf-8"))
                        if len(matches) >= 2:
                            break
                cursor.close()
                if len(matches) == 1:
                    return matches[0]
                return None

    def get_session_title(self, session_id: str) -> Optional[str]:
        """Get the title for a session, or None."""
        s = self.get_session(session_id)
        return s.get("title") if s else None

    def get_session_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Look up a session by exact title."""
        if not title:
            return None
        # Iterate all sessions and find by title (LMDB has no title index)
        with self._lock:
            with self._env.begin() as txn:
                cursor = txn.cursor(db=self._sessions_db)
                for key_bytes, value_bytes in cursor:
                    try:
                        session = json.loads(value_bytes.decode("utf-8"))
                        if session.get("title") == title:
                            cursor.close()
                            return session
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                cursor.close()
                return None

    def set_session_title(self, session_id: str, title: str) -> bool:
        """Set or update a session's title. Returns True if session found."""
        title = _sanitize_title(title)
        with self._lock:
            with self._env.begin(write=True) as txn:
                sdb = self._sessions_db
                session = _lmdb_get_json(txn, sdb, session_id)
                if session is None:
                    return False
                # Check uniqueness
                if title:
                    cursor = txn.cursor(db=sdb)
                    for key_bytes, value_bytes in cursor:
                        try:
                            other = json.loads(value_bytes.decode("utf-8"))
                            if other.get("title") == title and other.get("id") != session_id:
                                cursor.close()
                                raise ValueError(
                                    f"Title '{title}' is already in use by session {other['id']}"
                                )
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                    cursor.close()
                session["title"] = title
                _lmdb_put(txn, sdb, session_id, session)
                return True

    def prune_empty_ghost_sessions(self, sessions_dir: "Optional[Path]" = None) -> int:
        """Remove empty TUI ghost sessions (no messages, no title, >24hr old)."""
        cutoff = time.time() - 86400
        removed_ids: List[str] = []
        with self._lock:
            with self._env.begin(write=True) as txn:
                sdb = self._sessions_db
                mdb = self._messages_db
                cursor = txn.cursor(db=sdb)
                to_delete: List[str] = []
                for key_bytes, value_bytes in cursor:
                    try:
                        session = json.loads(value_bytes.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if (session.get("source") == "tui"
                            and session.get("title") is None
                            and session.get("ended_at") is not None
                            and (session.get("started_at") or 0) < cutoff
                            and _lmdb_count_prefix(txn, mdb, f"{session['id']}:") == 0):
                        to_delete.append(session["id"])
                cursor.close()
                for sid in to_delete:
                    _lmdb_delete(txn, sdb, sid)
                    self._remove_session_files(sessions_dir, sid)
                removed_ids = to_delete
        return len(removed_ids)

    def finalize_orphaned_compression_sessions(self) -> int:
        """Mark orphaned compression continuation sessions as ended."""
        cutoff = time.time() - 604800
        count = 0
        now = time.time()
        with self._lock:
            with self._env.begin(write=True) as txn:
                sdb = self._sessions_db
                # Find orphaned children: api_call_count=0, end_reason IS NULL,
                # started_at < cutoff, has parent_session_id,
                # parent exists with end_reason='compression', has messages
                cursor = txn.cursor(db=sdb)
                to_fix: List[str] = []
                for key_bytes, value_bytes in cursor:
                    try:
                        session = json.loads(value_bytes.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if (session.get("api_call_count") == 0
                            and session.get("end_reason") is None
                            and session.get("ended_at") is None
                            and (session.get("started_at") or 0) < cutoff
                            and session.get("parent_session_id") is not None):
                        # Check parent
                        parent = _lmdb_get_json(txn, sdb, session["parent_session_id"])
                        if parent and parent.get("end_reason") == "compression":
                            # Check has messages
                            msg_count = _lmdb_count_prefix(txn, self._messages_db, f"{session['id']}:")
                            if msg_count > 0:
                                to_fix.append(session["id"])
                cursor.close()
                for sid in to_fix:
                    session = _lmdb_get_json(txn, sdb, sid)
                    if session:
                        session["ended_at"] = now
                        session["end_reason"] = "orphaned_compression"
                        _lmdb_put(txn, sdb, sid, session)
                        count += 1
        return count

    def list_sessions_rich(
        self,
        source: str = None,
        exclude_sources: List[str] = None,
        limit: int = 20,
        offset: int = 0,
        include_children: bool = False,
        project_compression_tips: bool = True,
        order_by_last_active: bool = False,
    ) -> List[Dict[str, Any]]:
        """List sessions with preview and last_active, emulating the SQL version.

        Returns dicts with: id, source, model, title, started_at, ended_at,
        message_count, preview (first user message), last_active.
        """
        with self._lock:
            with self._env.begin() as txn:
                sdb = self._sessions_db
                mdb = self._messages_db
                all_sessions: List[Dict[str, Any]] = []

                cursor = txn.cursor(db=sdb)
                for key_bytes, value_bytes in cursor:
                    try:
                        session = json.loads(value_bytes.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue

                    # Filter: exclude children
                    if not include_children and session.get("parent_session_id"):
                        # Check if parent ended with 'branched' for exception
                        parent = _lmdb_get_json(txn, sdb, session["parent_session_id"])
                        if not (parent and parent.get("end_reason") == "branched"
                                and (session.get("started_at") or 0) >= (parent.get("ended_at") or 0)):
                            continue

                    # Filter by source
                    if source and session.get("source") != source:
                        continue

                    # Filter by exclude_sources
                    if exclude_sources and session.get("source") in exclude_sources:
                        continue

                    # Compute preview (first user message)
                    preview = ""
                    msg_prefix = f"{session['id']}:"
                    msg_cursor = txn.cursor(db=mdb)
                    if msg_cursor.set_range(msg_prefix.encode("utf-8")):
                        for mk, mv in msg_cursor:
                            if not mk.startswith(msg_prefix.encode("utf-8")):
                                break
                            try:
                                msg = json.loads(mv.decode("utf-8"))
                                if msg.get("role") == "user" and msg.get("content"):
                                    raw = msg["content"]
                                    if isinstance(raw, str):
                                        preview = raw.replace("\n", " ")[:63]
                                    break
                            except (UnicodeDecodeError, json.JSONDecodeError):
                                continue
                    msg_cursor.close()

                    # Compute last_active
                    last_active = session.get("started_at") or 0
                    last_msg_cursor = txn.cursor(db=mdb)
                    if last_msg_cursor.set_range(msg_prefix.encode("utf-8")):
                        last_ts = last_active
                        for mk, mv in last_msg_cursor:
                            if not mk.startswith(msg_prefix.encode("utf-8")):
                                break
                            try:
                                msg = json.loads(mv.decode("utf-8"))
                                ts = msg.get("timestamp") or 0
                                if ts > last_ts:
                                    last_ts = ts
                            except (UnicodeDecodeError, json.JSONDecodeError):
                                continue
                        if last_ts > last_active:
                            last_active = last_ts
                    last_msg_cursor.close()

                    row = dict(session)
                    row["preview"] = preview
                    row["last_active"] = last_active
                    all_sessions.append(row)

                cursor.close()

        # Sort
        if order_by_last_active:
            all_sessions.sort(key=lambda s: -(s.get("last_active") or s.get("started_at") or 0))
        else:
            all_sessions.sort(key=lambda s: -(s.get("started_at") or 0))

        # Apply offset + limit
        return all_sessions[offset:offset + limit]

    def search_sessions(
        self,
        query: str = None,
        source: str = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Search sessions by keyword (BM25) or list by source.

        If query is provided, uses BM25 full-text search.
        Otherwise lists sessions filtered by source.
        """
        if query:
            return self._search_by_keyword(query, limit=limit, source=source)
        return self._list_sessions(source=source, limit=limit, offset=offset)

    def _search_by_keyword(
        self,
        query: str,
        limit: int = 20,
        source: str = None,
    ) -> List[Dict[str, Any]]:
        """BM25 full-text search, return sessions."""
        if not self._bm25:
            return []
        bm25_results = self._bm25.search(query, limit=limit * 2)  # fetch extra for source filter
        sessions = []
        with self._lock:
            with self._env.begin() as txn:
                for r in bm25_results:
                    if len(sessions) >= limit:
                        break
                    doc_id = r.get("id", "")
                    session_id = doc_id.split(":msg:")[0] if ":msg:" in doc_id else doc_id
                    session = self.get_session(session_id)
                    if session is None:
                        continue
                    if source and session.get("source") != source:
                        continue
                    session["_search_score"] = r.get("score", 0.0)
                    session["content_preview"] = r.get("content_preview", "")
                    sessions.append(session)
                return sessions

    def _list_sessions(
        self,
        source: str = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        return self.list_sessions_rich(
            source=source,
            limit=limit,
            offset=offset,
            include_children=True,
            project_compression_tips=False,
            order_by_last_active=True,
        )

    # ── Counting ──

    def session_count(self, source: str = None) -> int:
        """Count sessions, optionally filtered by source."""
        with self._lock:
            with self._env.begin() as txn:
                sdb = self._sessions_db
                if source is None:
                    cursor = txn.cursor(db=sdb)
                    count = sum(1 for _ in cursor)
                    cursor.close()
                    return count
                count = 0
                cursor = txn.cursor(db=sdb)
                for key_bytes, value_bytes in cursor:
                    try:
                        session = json.loads(value_bytes.decode("utf-8"))
                        if session.get("source") == source:
                            count += 1
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                cursor.close()
                return count

    def message_count(self, session_id: str = None) -> int:
        """Count messages, optionally for a specific session."""
        with self._lock:
            with self._env.begin() as txn:
                if session_id:
                    return _lmdb_count_prefix(txn, self._messages_db, f"{session_id}:")
                mdb = self._messages_db
                count = 0
                cursor = txn.cursor(db=mdb)
                count = sum(1 for _ in cursor)
                cursor.close()
                return count

    # ── Export ──

    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export a single session with all its messages."""
        session = self.get_session(session_id)
        if not session:
            return None
        messages = self.get_messages(session_id)
        return {**session, "messages": messages}

    def export_all(self, source: str = None) -> List[Dict[str, Any]]:
        """Export all sessions with messages."""
        sessions = self.search_sessions(source=source, limit=100000)
        results = []
        for session in sessions:
            messages = self.get_messages(session["id"])
            results.append({**session, "messages": messages})
        return results

    # ── Messages ──

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str = None,
        tool_name: str = None,
        tool_calls: Any = None,
        tool_call_id: str = None,
        token_count: int = None,
        finish_reason: str = None,
        reasoning: str = None,
        reasoning_content: str = None,
        reasoning_details: Any = None,
        codex_reasoning_items: Any = None,
        codex_message_items: Any = None,
        platform_message_id: str = None,
        observed: bool = False,
    ) -> int:
        """Append a message to a session. Returns the message row ID."""
        # Pre-serialize structured fields
        reasoning_details_json = (
            json.dumps(reasoning_details) if reasoning_details else None
        )
        codex_items_json = (
            json.dumps(codex_reasoning_items) if codex_reasoning_items else None
        )
        codex_message_items_json = (
            json.dumps(codex_message_items) if codex_message_items else None
        )
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None
        stored_content = _encode_content(content)

        num_tool_calls = 0
        if tool_calls is not None:
            num_tool_calls = len(tool_calls) if isinstance(tool_calls, list) else 1

        with self._lock:
            msg_id = None
            with self._env.begin(write=True) as txn:
                msg_id = self._next_msg_id(txn)

                msg = {
                    "id": msg_id,
                    "session_id": session_id,
                    "role": role,
                    "content": stored_content,
                    "tool_call_id": tool_call_id,
                    "tool_calls": tool_calls_json,
                    "tool_name": tool_name,
                    "timestamp": time.time(),
                    "token_count": token_count,
                    "finish_reason": finish_reason,
                    "reasoning": reasoning,
                    "reasoning_content": reasoning_content,
                    "reasoning_details": reasoning_details_json,
                    "codex_reasoning_items": codex_items_json,
                    "codex_message_items": codex_message_items_json,
                    "platform_message_id": platform_message_id,
                    "observed": 1 if observed else 0,
                }

                # Store message: key = session_id:msg_id
                msg_key = self._msg_key(session_id, msg_id)
                _lmdb_put(txn, self._messages_db, msg_key, msg)

                # Store index: msg_id -> session_id
                _lmdb_put(txn, self._msg_idx_db, str(msg_id), {"session_id": session_id})

                # Update session counters
                sdb = self._sessions_db
                session = _lmdb_get_json(txn, sdb, session_id)
                if session:
                    session["message_count"] = (session.get("message_count") or 0) + 1
                    if num_tool_calls > 0:
                        session["tool_call_count"] = (session.get("tool_call_count") or 0) + num_tool_calls
                    _lmdb_put(txn, sdb, session_id, session)

                # Add to BM25 index (skip if BM25 unavailable)
                search_text = " ".join(filter(None, [
                    content if isinstance(content, str) else "",
                    tool_name or "",
                    tool_calls_json or "",
                ]))
                if search_text.strip() and self._bm25:
                    self._bm25.add(
                        f"{session_id}:msg:{msg_id}",
                        search_text[:_BM25_MAX_TEXT_LENGTH],
                        {"session_id": session_id, "role": role},
                    )

            # Async-save BM25 (if loaded)
            self._write_bm25_async()

            return msg_id

    def _decode_message_row(self, row: dict) -> dict:
        """Decode a message dict (same as SQLite version's post-processing)."""
        msg = dict(row)
        if "content" in msg:
            msg["content"] = _decode_content(msg["content"])
        if msg.get("tool_calls"):
            try:
                msg["tool_calls"] = json.loads(msg["tool_calls"])
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to deserialize tool_calls, falling back to []")
                msg["tool_calls"] = []
        return msg

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Load all messages for a session, ordered by insertion order."""
        if not session_id:
            return []
        with self._lock:
            with self._env.begin() as txn:
                entries = _lmdb_iter_prefix(txn, self._messages_db, f"{session_id}:")
                # Sort by msg_id (from key)
                def _sort_key(e):
                    parsed = self._parse_msg_key(e[0])
                    return parsed[1] if parsed else 0
                entries.sort(key=_sort_key)
                return [self._decode_message_row(e[1]) for e in entries]

    def get_messages_around(
        self,
        session_id: str,
        around_message_id: int,
        window: int = 5,
    ) -> Dict[str, Any]:
        """Load a window of messages anchored on a specific message id.

        Returns {"window": [...], "messages_before": N, "messages_after": N}.
        """
        if window < 0:
            window = 0
        if not session_id:
            return {"window": [], "messages_before": 0, "messages_after": 0}

        with self._lock:
            with self._env.begin() as txn:
                mdb = self._messages_db

                # Find the anchor message
                anchor = None
                entries = _lmdb_iter_prefix(txn, mdb, f"{session_id}:")
                for k, v in entries:
                    parsed = self._parse_msg_key(k)
                    if parsed and parsed[1] == around_message_id:
                        anchor = (k, v)
                        break

                if anchor is None:
                    return {"window": [], "messages_before": 0, "messages_after": 0}

                # Sort by msg_id
                def _ekey(e):
                    p = self._parse_msg_key(e[0])
                    return p[1] if p else 0
                entries.sort(key=_ekey)

                # Find anchor index
                anchor_idx = None
                for i, (k, _) in enumerate(entries):
                    p = self._parse_msg_key(k)
                    if p and p[1] == around_message_id:
                        anchor_idx = i
                        break

                if anchor_idx is None:
                    return {"window": [], "messages_before": 0, "messages_after": 0}

                # Extract window
                start = max(0, anchor_idx - window)
                end = min(len(entries), anchor_idx + window + 1)
                window_msgs = [self._decode_message_row(v) for _, v in entries[start:end]]

                return {
                    "window": window_msgs,
                    "messages_before": anchor_idx - start,
                    "messages_after": end - anchor_idx - 1,
                }

    def get_anchored_view(
        self,
        session_id: str,
        around_message_id: int,
        window: int = 5,
        bookend: int = 3,
        keep_roles: Optional[Tuple[str, ...]] = ("user", "assistant"),
    ) -> Dict[str, Any]:
        """Return an anchored window plus session bookends.

        Same API as the SQLite version's get_anchored_view.
        """
        if bookend < 0:
            bookend = 0

        primitive = self.get_messages_around(session_id, around_message_id, window=window)
        window_rows = primitive["window"]
        if not window_rows:
            return {
                "window": [],
                "messages_before": 0,
                "messages_after": 0,
                "bookend_start": [],
                "bookend_end": [],
            }

        # Apply role filter to window (but never drop anchor)
        if keep_roles is not None:
            keep_set = set(keep_roles)
            filtered_window = [
                m for m in window_rows
                if m.get("id") == around_message_id or m.get("role") in keep_set
            ]
        else:
            filtered_window = window_rows

        window_min_id = window_rows[0]["id"]
        window_max_id = window_rows[-1]["id"]

        bookend_start_rows: List[Dict[str, Any]] = []
        bookend_end_rows: List[Dict[str, Any]] = []

        # Fetch bookends
        all_msgs = self.get_messages(session_id)
        if all_msgs:
            # Bookend start: first N user/assistant msgs before window
            before_window = [m for m in all_msgs
                             if m["id"] < window_min_id
                             and (keep_roles is None or m.get("role") in keep_set)
                             and (m.get("content") or "").strip()]
            bookend_start_rows = before_window[:bookend]

            # Bookend end: last N user/assistant msgs after window
            after_window = [m for m in all_msgs
                            if m["id"] > window_max_id
                            and (keep_roles is None or m.get("role") in keep_set)
                            and (m.get("content") or "").strip()]
            bookend_end_rows = after_window[-bookend:] if after_window else []

        return {
            "window": filtered_window,
            "messages_before": primitive.get("messages_before", 0),
            "messages_after": primitive.get("messages_after", 0),
            "bookend_start": bookend_start_rows,
            "bookend_end": bookend_end_rows,
        }

    def get_messages_as_conversation(
        self, session_id: str, include_ancestors: bool = False
    ) -> List[Dict[str, Any]]:
        """Load messages in OpenAI conversation format."""
        from agent.memory_manager import sanitize_context

        session_ids = [session_id]
        if include_ancestors:
            session_ids = self._session_lineage_root_to_tip(session_id)

        all_msgs: List[Dict[str, Any]] = []
        for sid in session_ids:
            all_msgs.extend(self.get_messages(sid))

        messages = []
        seen_platform_ids = set()
        for row in all_msgs:
            content = _decode_content(row.get("content"))
            if row["role"] in {"user", "assistant"} and isinstance(content, str):
                content = sanitize_context(content).strip()
            msg = {"role": row["role"], "content": content}

            if row.get("tool_call_id"):
                msg["tool_call_id"] = row["tool_call_id"]
            if row.get("tool_name"):
                msg["tool_name"] = row["tool_name"]
            if row.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(row["tool_calls"]) if isinstance(row["tool_calls"], str) else row["tool_calls"]
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Failed to deserialize tool_calls in conversation replay, falling back to []")
                    msg["tool_calls"] = []

            if row.get("platform_message_id"):
                msg["message_id"] = row["platform_message_id"]

            if row.get("observed"):
                msg["observed"] = True

            if row["role"] == "assistant":
                if row.get("finish_reason"):
                    msg["finish_reason"] = row["finish_reason"]
                if row.get("reasoning"):
                    msg["reasoning"] = row["reasoning"]
                if row.get("reasoning_content") is not None:
                    msg["reasoning_content"] = row["reasoning_content"]
                if row.get("reasoning_details"):
                    try:
                        msg["reasoning_details"] = json.loads(row["reasoning_details"]) if isinstance(row["reasoning_details"], str) else row["reasoning_details"]
                    except (json.JSONDecodeError, TypeError):
                        msg["reasoning_details"] = None
                if row.get("codex_reasoning_items"):
                    try:
                        msg["codex_reasoning_items"] = json.loads(row["codex_reasoning_items"]) if isinstance(row["codex_reasoning_items"], str) else row["codex_reasoning_items"]
                    except (json.JSONDecodeError, TypeError):
                        msg["codex_reasoning_items"] = None
                if row.get("codex_message_items"):
                    try:
                        msg["codex_message_items"] = json.loads(row["codex_message_items"]) if isinstance(row["codex_message_items"], str) else row["codex_message_items"]
                    except (json.JSONDecodeError, TypeError):
                        msg["codex_message_items"] = None

            if include_ancestors and self._is_duplicate_replayed_user_message(messages, msg):
                continue
            messages.append(msg)

        return messages

    def _session_lineage_root_to_tip(self, session_id: str) -> List[str]:
        """Walk parent_session_id chain to root, then return root-to-tip."""
        if not session_id:
            return [session_id]

        # Walk backwards to root
        chain = []
        current = session_id
        seen = set()
        for _ in range(100):
            if not current or current in seen:
                break
            seen.add(current)
            chain.append(current)
            session = self.get_session(current)
            if session is None:
                break
            parent = session.get("parent_session_id")
            if not parent:
                break
            current = parent
        return list(reversed(chain)) or [session_id]

    @staticmethod
    def _is_duplicate_replayed_user_message(
        messages: List[Dict[str, Any]],
        candidate: Dict[str, Any],
    ) -> bool:
        """Check if a user message is a duplicate of the last replayed one."""
        if candidate.get("role") != "user":
            return False
        for m in reversed(messages):
            if m.get("role") == "user":
                return (m.get("content") == candidate.get("content")
                        and m.get("tool_call_id") == candidate.get("tool_call_id"))
        return False

    def replace_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Atomically replace every message for a session."""
        with self._lock:
            with self._env.begin(write=True) as txn:
                mdb = self._messages_db
                midx = self._msg_idx_db
                sdb = self._sessions_db

                # Remove old messages for this session
                old_entries = _lmdb_iter_prefix(txn, mdb, f"{session_id}:")
                for k, v in old_entries:
                    parsed = self._parse_msg_key(k)
                    if parsed:
                        old_mid = str(parsed[1])
                        _lmdb_delete(txn, mdb, k)
                        _lmdb_delete(txn, midx, old_mid)
                        self._bm25.remove(old_mid)

                # Insert new messages
                for msg in messages:
                    msg_id = self._next_msg_id(txn)
                    msg_key = self._msg_key(session_id, msg_id)
                    msg_copy = dict(msg)
                    msg_copy["id"] = msg_id
                    msg_copy["session_id"] = session_id
                    if "timestamp" not in msg_copy:
                        msg_copy["timestamp"] = time.time()

                    # Encode content
                    raw_content = msg_copy.get("content")
                    if isinstance(raw_content, (list, dict)):
                        msg_copy["content"] = json.dumps(raw_content, ensure_ascii=False, default=str)
                    elif raw_content is not None:
                        msg_copy["content"] = str(raw_content)

                    # Encode tool_calls
                    tc = msg_copy.get("tool_calls")
                    if tc and not isinstance(tc, str):
                        msg_copy["tool_calls"] = json.dumps(tc, ensure_ascii=False, default=str)

                    _lmdb_put(txn, mdb, msg_key, msg_copy)
                    _lmdb_put(txn, midx, str(msg_id), {"session_id": session_id})

                    # BM25 index
                    search_text = " ".join(filter(None, [
                        str(msg_copy.get("content", "")),
                        str(msg_copy.get("tool_name", "")),
                        str(msg_copy.get("tool_calls", "")),
                    ]))
                    if search_text.strip() and self._bm25:
                        self._bm25.add(
                            f"{session_id}:msg:{msg_id}",
                            search_text[:_BM25_MAX_TEXT_LENGTH],
                            {"session_id": session_id, "role": msg_copy.get("role", "")},
                        )

                # Update session message count
                session = _lmdb_get_json(txn, sdb, session_id)
                if session:
                    session["message_count"] = len(messages)
                    tc_count = sum(
                        1 for m in messages
                        if m.get("role") == "tool" or m.get("tool_calls")
                    )
                    session["tool_call_count"] = tc_count
                    _lmdb_put(txn, sdb, session_id, session)

            if self._bm25:
                self._write_bm25_async()

    def clear_messages(self, session_id: str) -> None:
        """Delete all messages for a session and reset its counters."""
        with self._lock:
            with self._env.begin(write=True) as txn:
                mdb = self._messages_db
                midx = self._msg_idx_db
                sdb = self._sessions_db

                old_entries = _lmdb_iter_prefix(txn, mdb, f"{session_id}:")
                for k, v in old_entries:
                    parsed = self._parse_msg_key(k)
                    if parsed:
                        old_mid = str(parsed[1])
                        _lmdb_delete(txn, mdb, k)
                        _lmdb_delete(txn, midx, old_mid)
                        if self._bm25:
                            self._bm25.remove(old_mid)

                session = _lmdb_get_json(txn, sdb, session_id)
                if session:
                    session["message_count"] = 0
                    session["tool_call_count"] = 0
                    _lmdb_put(txn, sdb, session_id, session)

            if self._bm25:
                self._write_bm25_async()

    # ── Search ──

    def search_messages(
        self,
        query: str,
        source_filter: List[str] = None,
        exclude_sources: List[str] = None,
        role_filter: List[str] = None,
        limit: int = 20,
        offset: int = 0,
        sort: str = None,
    ) -> List[Dict[str, Any]]:
        """Full-text search across session messages using BM25.

        Supports FTS5-compatible syntax (AND/OR/NOT) adapted for BM25.
        Returns matching messages with session metadata and content snippet.
        """
        if not query or not query.strip():
            return []
        if not self._bm25:
            return []

        # Sanitize and convert FTS5 query to BM25
        bm25_query = _sanitize_fts5_query(query)
        bm25_query = _fts5_to_bm25_query(bm25_query)
        if not bm25_query:
            return []

        # Run BM25 search with generous limit for post-filtering
        results = self._bm25.search(bm25_query, limit=limit + offset + 200)
        if not results:
            return []

        with self._lock:
            with self._env.begin() as txn:
                sdb = self._sessions_db
                mdb = self._messages_db

                matches = []
                for r in results:
                    doc_id = r["id"]
                    # Doc ID format: "{session_id}:msg:{msg_id}"
                    if ":msg:" in doc_id:
                        parts = doc_id.split(":msg:")
                        session_id = parts[0]
                        try:
                            msg_id = int(parts[1])
                        except (ValueError, IndexError):
                            continue
                    else:
                        session_id = r.get("session_id") or doc_id
                        try:
                            msg_id = int(doc_id)
                        except ValueError:
                            continue

                    # Get session metadata for filtering
                    session = _lmdb_get_json(txn, sdb, session_id)

                    # Source filter
                    if session:
                        if source_filter and session.get("source") not in source_filter:
                            continue
                        if exclude_sources and session.get("source") in exclude_sources:
                            continue

                    # Get the actual message
                    msg_key = self._msg_key(session_id, msg_id)
                    msg_raw = _lmdb_get_json(txn, mdb, msg_key)

                    if msg_raw is None:
                        continue

                    msg = self._decode_message_row(msg_raw)

                    # Role filter
                    if role_filter and msg.get("role") not in role_filter:
                        continue

                    # Build snippet (like FTS5 snippet function)
                    content_str = msg.get("content") or ""
                    if isinstance(content_str, list):
                        text_parts = [p.get("text", "") for p in content_str
                                      if isinstance(p, dict) and p.get("type") == "text"]
                        content_str = " ".join(text_parts)
                    elif not isinstance(content_str, str):
                        content_str = str(content_str)

                    snippet = content_str[:120] + "..." if len(content_str) > 120 else content_str

                    entry = {
                        "id": msg_id,
                        "session_id": session_id,
                        "role": msg.get("role"),
                        "snippet": snippet,
                        "content": content_str,
                        "timestamp": msg.get("timestamp"),
                        "tool_name": msg.get("tool_name"),
                        "source": session.get("source") if session else "unknown",
                        "model": session.get("model") if session else None,
                        "session_started": session.get("started_at") if session else None,
                    }
                    matches.append(entry)

        # Sort
        if sort == "newest":
            matches.sort(key=lambda m: -(m.get("timestamp") or 0))
        elif sort == "oldest":
            matches.sort(key=lambda m: m.get("timestamp") or 0)
        else:
            # Relevance-order — already ranked by BM25 score
            pass

        # Apply offset + limit
        return matches[offset:offset + limit]

    # ── Session deletion ──

    @staticmethod
    def _remove_session_files(sessions_dir: Optional[Path], session_id: str) -> None:
        """Remove on-disk transcript files for a session."""
        if sessions_dir is None:
            return
        for suffix in (".json", ".jsonl"):
            p = sessions_dir / f"{session_id}{suffix}"
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            for p in sessions_dir.glob(f"request_dump_{session_id}_*.json"):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
        except OSError:
            pass

    def delete_session(
        self,
        session_id: str,
        sessions_dir: Optional[Path] = None,
    ) -> bool:
        """Delete a session and all its messages.

        Child sessions are orphaned (parent_session_id set to NULL) rather
        than cascade-deleted.
        """
        with self._lock:
            with self._env.begin(write=True) as txn:
                sdb = self._sessions_db
                session = _lmdb_get_json(txn, sdb, session_id)
                if session is None:
                    return False

                # Orphan child sessions
                cursor = txn.cursor(db=sdb)
                to_orphan: List[str] = []
                for key_bytes, value_bytes in cursor:
                    try:
                        child = json.loads(value_bytes.decode("utf-8"))
                        if child.get("parent_session_id") == session_id:
                            to_orphan.append(child["id"])
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                cursor.close()
                for cid in to_orphan:
                    child = _lmdb_get_json(txn, sdb, cid)
                    if child:
                        child["parent_session_id"] = None
                        _lmdb_put(txn, sdb, cid, child)

                # Delete messages
                msg_entries = _lmdb_iter_prefix(txn, self._messages_db, f"{session_id}:")
                for k, v in msg_entries:
                    parsed = self._parse_msg_key(k)
                    if parsed:
                        old_mid = str(parsed[1])
                        _lmdb_delete(txn, self._messages_db, k)
                        _lmdb_delete(txn, self._msg_idx_db, old_mid)
                        if self._bm25:
                            self._bm25.remove(old_mid)

                # Delete session
                _lmdb_delete(txn, sdb, session_id)

            if self._bm25:
                self._write_bm25_async()

        self._remove_session_files(sessions_dir, session_id)
        return True

    def prune_sessions(
        self,
        older_than_days: int = 90,
        source: str = None,
        sessions_dir: Optional[Path] = None,
    ) -> int:
        """Delete ended sessions older than N days. Returns count of deleted sessions."""
        cutoff = time.time() - (older_than_days * 86400)
        removed_ids: List[str] = []

        with self._lock:
            with self._env.begin(write=True) as txn:
                sdb = self._sessions_db
                cursor = txn.cursor(db=sdb)
                to_delete: List[str] = []
                for key_bytes, value_bytes in cursor:
                    try:
                        session = json.loads(value_bytes.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if ((session.get("started_at") or 0) < cutoff
                            and session.get("ended_at") is not None
                            and (source is None or session.get("source") == source)):
                        to_delete.append(session["id"])
                cursor.close()

                for sid in to_delete:
                    # Orphan children
                    c_cursor = txn.cursor(db=sdb)
                    for ck_bytes, cv_bytes in c_cursor:
                        try:
                            child = json.loads(cv_bytes.decode("utf-8"))
                            if child.get("parent_session_id") == sid:
                                child["parent_session_id"] = None
                                _lmdb_put(txn, sdb, child["id"], child)
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                    c_cursor.close()

                    # Delete messages
                    msg_entries = _lmdb_iter_prefix(txn, self._messages_db, f"{sid}:")
                    for k, v in msg_entries:
                        parsed = self._parse_msg_key(k)
                        if parsed:
                            old_mid = str(parsed[1])
                            _lmdb_delete(txn, self._messages_db, k)
                            _lmdb_delete(txn, self._msg_idx_db, old_mid)
                            if self._bm25:
                                self._bm25.remove(old_mid)

                    _lmdb_delete(txn, sdb, sid)
                    removed_ids.append(sid)

            if self._bm25:
                self._write_bm25_async()

        for sid in removed_ids:
            self._remove_session_files(sessions_dir, sid)
        return len(removed_ids)

    # ── Meta key/value ──

    def get_meta(self, key: str) -> Optional[str]:
        """Read a value from the meta key/value store."""
        with self._lock:
            with self._env.begin() as txn:
                raw = txn.get(key.encode("utf-8"), db=self._meta_db)
                if raw is None:
                    return None
                return raw.decode("utf-8")

    def set_meta(self, key: str, value: str) -> None:
        """Write a value to the meta key/value store."""
        with self._lock:
            with self._env.begin(write=True) as txn:
                txn.put(key.encode("utf-8"), value.encode("utf-8"), db=self._meta_db)

    # ── Telegram topic migration (stub) ──

    def apply_telegram_topic_migration(self) -> None:
        """Stub: Telegram topic migration was SQLite-specific. No-op in LMDB."""
        pass

    # ── Handoff management ──

    def _update_handoff_field(self, session_id: str, field: str, value: Any) -> None:
        """Update a single handoff-related field on a session."""
        with self._lock:
            with self._env.begin(write=True) as txn:
                sdb = self._sessions_db
                session = _lmdb_get_json(txn, sdb, session_id)
                if session:
                    session[field] = value
                    _lmdb_put(txn, sdb, session_id, session)

    def set_handoff(self, session_id: str, handoff_platform: str) -> None:
        """Set handoff platform on a session."""
        self._update_handoff_field(session_id, "handoff_platform", handoff_platform)

    def set_handoff_state(self, session_id: str, state: str) -> None:
        """Set handoff state on a session."""
        self._update_handoff_field(session_id, "handoff_state", state)

    def list_pending_handoffs(self) -> List[Dict[str, Any]]:
        """List all sessions with handoff_state='pending'."""
        result = []
        with self._lock:
            with self._env.begin() as txn:
                sdb = self._sessions_db
                cursor = txn.cursor(db=sdb)
                for key_bytes, value_bytes in cursor:
                    try:
                        session = json.loads(value_bytes.decode("utf-8"))
                        if session.get("handoff_state") == "pending":
                            result.append(session)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                cursor.close()
        return result

    def claim_handoff(self, session_id: str) -> bool:
        """Atomically transition pending → running. Returns True if claimed."""
        with self._lock:
            with self._env.begin(write=True) as txn:
                sdb = self._sessions_db
                session = _lmdb_get_json(txn, sdb, session_id)
                if session and session.get("handoff_state") == "pending":
                    session["handoff_state"] = "running"
                    _lmdb_put(txn, sdb, session_id, session)
                    return True
                return False

    def complete_handoff(self, session_id: str) -> None:
        """Mark a handoff as completed."""
        self._update_handoff_field(session_id, "handoff_state", "completed")

    def fail_handoff(self, session_id: str, error: str) -> None:
        """Mark a handoff as failed and record the reason."""
        with self._lock:
            with self._env.begin(write=True) as txn:
                sdb = self._sessions_db
                session = _lmdb_get_json(txn, sdb, session_id)
                if session:
                    session["handoff_state"] = "failed"
                    session["handoff_error"] = error[:500]
                    _lmdb_put(txn, sdb, session_id, session)

    # ── Compression ──

    def get_compression_tip(self, session_id: str) -> str:
        """Walk compression chain forward to the live tip."""
        visited = set()
        current = session_id
        with self._lock:
            with self._env.begin() as txn:
                sdb = self._sessions_db
                while current and current not in visited:
                    visited.add(current)
                    session = _lmdb_get_json(txn, sdb, current)
                    if session is None:
                        break
                    parent = session.get("parent_session_id")
                    if parent and parent in visited:
                        break
                    # Check if this session has a compression child
                    cursor = txn.cursor(db=sdb)
                    found_child = None
                    for key_bytes, value_bytes in cursor:
                        try:
                            child = json.loads(value_bytes.decode("utf-8"))
                            if (child.get("parent_session_id") == current
                                    and child.get("started_at", 0) >= (session.get("ended_at") or 0)):
                                found_child = child["id"]
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                    cursor.close()
                    if found_child:
                        current = found_child
                    else:
                        break
        return current

    def get_root_session_id(self, session_id: str) -> str:
        """Walk parent_session_id chain to the lineage root."""
        visited = set()
        current = session_id
        with self._lock:
            with self._env.begin() as txn:
                sdb = self._sessions_db
                while current and current not in visited:
                    visited.add(current)
                    session = _lmdb_get_json(txn, sdb, current)
                    if session is None:
                        break
                    parent = session.get("parent_session_id")
                    if not parent or parent in visited:
                        break
                    current = parent
        return current

    # ── Maintenance ──

    def close(self) -> None:
        """Close the LMDB environment and save BM25 index."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._bm25 and self._bm25.is_dirty:
                self._bm25.save(str(self.bm25_path))
        except Exception as exc:
            logger.warning("BM25 save on close failed: %s", exc)
        if self._env:
            try:
                _release_lmdb_env(str(self.db_path))
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ── __all__ ──
# Keep the import symbols that consumer code uses.
# Consumers import: SessionDB, DEFAULT_DB_PATH, SCHEMA_VERSION,
#   get_last_init_error, format_session_db_unavailable,
#   apply_wal_with_fallback, _set_last_init_error, _WAL_INCOMPAT_MARKERS
#   sanitize_title (via SessionDB.sanitize_title, which is the classmethod)
SessionDB.sanitize_title = staticmethod(_sanitize_title)
