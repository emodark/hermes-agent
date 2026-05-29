#!/usr/bin/env python3
"""从 LMDB 中重建 BM25 全文搜索索引。"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hermes_state import SessionDB
from hermes_bm25 import BM25Index

db = SessionDB()
print(f"Sessions: {db.session_count()}, Messages: {db.message_count()}")

bm25 = BM25Index()
t0 = time.time()
total = 0

sessions = db.list_sessions_rich(limit=5000)
print(f"Processing {len(sessions)} sessions...")

for si, s in enumerate(sessions):
    sid = s["id"]
    try:
        msgs = db.get_messages(sid)
    except Exception as e:
        print(f"  WARN: get_messages({sid}) failed: {e}")
        continue
    for m in msgs:
        content = m.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = str(content)
        content = content[:500]

        tool_name = m.get("tool_name", "") or ""
        role = m.get("role", "")

        txt = f"{content} {tool_name}".strip()
        if txt:
            doc_id = f'{sid}:msg:{m.get("id", total)}'
            bm25.add(doc_id, txt, {"session_id": sid, "role": role})
            total += 1

    if (si + 1) % 200 == 0:
        elapsed = time.time() - t0
        print(f"  [{elapsed:.0f}s] session {si+1}/{len(sessions)}, msgs={total}")

print(f"\nTotal BM25 docs added: {total}")
print(f"Saving to {db.bm25_path}...")
bm25.save(str(db.bm25_path))
elapsed = time.time() - t0
print(f"Done! {elapsed:.1f}s")

saved_size = os.path.getsize(str(db.bm25_path))
print(f"BM25 file size: {saved_size} bytes ({saved_size/1024:.0f} KB / {saved_size/1024/1024:.1f} MB)")

# Quick verification
print("\nVerification search:")
for q in ["gateway", "迁移", "测试", "股票", "hello"]:
    rr = bm25.search(q, limit=2)
    print(f"  search('{q}'): {len(rr)} results" + (f" - top={rr[0]['content_preview'][:60]}" if rr else ""))

print("\nDone!")
