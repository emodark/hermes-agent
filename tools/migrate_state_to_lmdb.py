#!/usr/bin/env python3
"""
迁移脚本：从 SQLite state.db 迁移到 LMDB state.lmdb + BM25。

用法：
  python3 tools/migrate_state_to_lmdb.py [--db-path /path/to/state.db]

默认读取 HERMES_HOME/state.db，写入同目录的 state.lmdb。
流式读取，不一次性加载全部消息到内存。
"""
import argparse
import json
import logging
import shutil
import sqlite3
import sys
import time
from pathlib import Path

_HERMES_AGENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERMES_AGENT))

import lmdb

from hermes_bm25 import BM25Index
from hermes_constants import get_hermes_home

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("migrate")

_BM25_MAX_TEXT_LENGTH = 20000
_BATCH_SIZE = 500
_LMDB_MAP_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
_LMDB_MAX_DBS = 32


def migrate(sqlite_path: Path, lmdb_path: Path, bm25_path: Path) -> None:
    """从 SQLite state.db 流式迁移数据到 LMDB + BM25。"""

    # ── 1. 删除旧的 LMDB ──
    if lmdb_path.exists():
        shutil.rmtree(lmdb_path)
        logger.info("Removed old LMDB: %s", lmdb_path)

    # ── 2. 连接 SQLite ──
    if not sqlite_path.exists():
        logger.error("SQLite DB not found: %s", sqlite_path)
        sys.exit(1)

    logger.info("Opening SQLite: %s", sqlite_path)
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row

    # 统计总数
    cursor = conn.execute("SELECT COUNT(*) as c FROM sessions")
    total_sessions = cursor.fetchone()["c"]
    cursor = conn.execute("SELECT COUNT(*) as c FROM messages")
    total_messages = cursor.fetchone()["c"]
    logger.info("Total: %d sessions, %d messages", total_sessions, total_messages)

    # ── 3. 打开 LMDB ──
    logger.info("Opening LMDB: %s (map_size=%dMB)", lmdb_path, _LMDB_MAP_SIZE // (1024 * 1024))
    env = lmdb.open(str(lmdb_path), map_size=_LMDB_MAP_SIZE, max_dbs=_LMDB_MAX_DBS)

    sessions_db = env.open_db(key=b"sessions", create=True)
    messages_db = env.open_db(key=b"messages", create=True)
    msg_idx_db = env.open_db(key=b"msg_idx", create=True)
    meta_db = env.open_db(key=b"meta", create=True)

    bm25 = BM25Index()

    # ── 4. 迁移 sessions（一次全部加载，2516条很小）──
    logger.info("Migrating sessions...")
    cursor = conn.execute("SELECT * FROM sessions")
    sessions_raw = [dict(row) for row in cursor.fetchall()]

    t0 = time.time()
    written_sessions = 0
    for i in range(0, len(sessions_raw), _BATCH_SIZE):
        batch = sessions_raw[i : i + _BATCH_SIZE]
        with env.begin(write=True) as txn:
            for session in batch:
                sid = session.get("id", "")
                if not sid:
                    continue
                encoded = json.dumps(session, ensure_ascii=False, default=str).encode("utf-8")
                txn.put(sid.encode("utf-8"), encoded, db=sessions_db)
                written_sessions += 1
    elapsed = time.time() - t0
    logger.info("Sessions written: %d/%d (%.1fs)", written_sessions, total_sessions, elapsed)

    # ── 5. 迁移 meta ──
    cursor = conn.execute("SELECT * FROM state_meta")
    meta_rows = {row["key"]: row["value"] for row in cursor.fetchall()}

    cursor = conn.execute("SELECT MAX(id) as max_id FROM messages")
    max_msg_id = cursor.fetchone()["max_id"] or 0

    with env.begin(write=True) as txn:
        txn.put(b"next_msg_id", str(max_msg_id + 1).encode(), db=meta_db)
        txn.put(b"schema_version", b"14", db=meta_db)
        for key, value in meta_rows.items():
            txn.put(key.encode("utf-8"), value.encode("utf-8"), db=meta_db)
    logger.info("Meta written (next_msg_id=%d)", max_msg_id + 1)

    # ── 6. 流式迁移 messages + BM25 ──
    logger.info("Migrating messages (streaming, batch=%d)...", _BATCH_SIZE)
    t0 = time.time()
    written_messages = 0
    last_log_time = t0
    offset = 0

    while offset < total_messages:
        cursor = conn.execute(
            "SELECT * FROM messages ORDER BY id LIMIT ? OFFSET ?",
            (_BATCH_SIZE, offset),
        )
        batch = [dict(row) for row in cursor.fetchall()]
        offset += _BATCH_SIZE

        # 写入 LMDB
        with env.begin(write=True) as txn:
            for msg in batch:
                msg_id = msg.get("id", 0)
                session_id = msg.get("session_id", "")
                if not session_id:
                    continue
                msg_key = f"{session_id}:{msg_id}"
                encoded = json.dumps(msg, ensure_ascii=False, default=str).encode("utf-8")
                txn.put(msg_key.encode("utf-8"), encoded, db=messages_db)
                idx_encoded = json.dumps({"session_id": session_id}, ensure_ascii=False).encode("utf-8")
                txn.put(str(msg_id).encode("utf-8"), idx_encoded, db=msg_idx_db)
                written_messages += 1

        # BM25 索引（内存中）
        for msg in batch:
            msg_id = msg.get("id", 0)
            sid = msg.get("session_id", "")
            content = msg.get("content") or ""
            tool_name = msg.get("tool_name") or ""
            tool_calls = msg.get("tool_calls") or ""
            if isinstance(tool_calls, (list, dict)):
                tool_calls = json.dumps(tool_calls, ensure_ascii=False)
            elif not isinstance(tool_calls, str):
                tool_calls = str(tool_calls)

            search_text = f"{content} {tool_name} {tool_calls}"
            if search_text.strip():
                bm25.add(f"{sid}:msg:{msg_id}", search_text[:_BM25_MAX_TEXT_LENGTH], {
                    "session_id": sid,
                    "role": msg.get("role", ""),
                })

        # 进度日志（每5秒至少一次）
        now = time.time()
        if now - last_log_time >= 5:
            rate = written_messages / (now - t0)
            eta = (total_messages - written_messages) / rate if rate > 0 else 0
            logger.info("  Messages: %d/%d (%.1f msg/s, ETA %.0fs)",
                        written_messages, total_messages, rate, eta)
            last_log_time = now

    elapsed = time.time() - t0
    logger.info("Messages written: %d/%d (%.1fs, avg %.0f msg/s)",
                written_messages, total_messages, elapsed,
                written_messages / elapsed if elapsed > 0 else 0)

    # ── 7. 保存 BM25 ──
    logger.info("Saving BM25 index (%d docs)...", bm25.doc_count())
    t0 = time.time()
    bm25.save(str(bm25_path))
    logger.info("BM25 saved to %s (%.1fs)", bm25_path, time.time() - t0)

    env.close()
    conn.close()

    logger.info("=== Migration complete! ===")
    logger.info("  Sessions: %d/%d", written_sessions, total_sessions)
    logger.info("  Messages: %d/%d", written_messages, total_messages)
    logger.info("  BM25 docs: %d", bm25.doc_count())


def main():
    parser = argparse.ArgumentParser(description="Migrate state.db (SQLite) to state.lmdb (LMDB + BM25)")
    parser.add_argument("--db-path", help="Path to SQLite state.db (default: HERMES_HOME/state.db)")
    args = parser.parse_args()

    if args.db_path:
        sqlite_path = Path(args.db_path)
    else:
        sqlite_path = get_hermes_home() / "state.db"

    lmdb_path = sqlite_path.with_suffix(".lmdb")
    bm25_path = lmdb_path.with_suffix(".lmdb.bm25.gz")

    migrate(sqlite_path, lmdb_path, bm25_path)


if __name__ == "__main__":
    main()
