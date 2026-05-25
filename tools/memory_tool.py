#!/usr/bin/env python3
"""
Memory Tool Module - Persistent Curated Memory

Provides bounded, file-backed memory that persists across sessions. Two stores:
  - MEMORY.md: agent's personal notes and observations (environment facts, project
    conventions, tool quirks, things learned)
  - USER.md: what the agent knows about the user (preferences, communication style,
    expectations, workflow habits)

Both are injected into the system prompt as a frozen snapshot at session start.
Mid-session writes update files on disk immediately (durable) but do NOT change
the system prompt -- this preserves the prefix cache for the entire session.
The snapshot refreshes on the next session start.

Entry delimiter: § (section sign). Entries can be multiline.
Character limits (not tokens) because char counts are model-independent.

Design:
- Single `memory` tool with action parameter: add, replace, remove, read
- replace/remove use short unique substring matching (not full text or IDs)
- Behavioral guidance lives in the tool schema description
- Frozen snapshot pattern: system prompt is stable, tool responses show live state
"""

import json
import hashlib
import logging
import os
import re
import subprocess
import tempfile
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Dict, Any, List, Optional

from utils import atomic_replace

# fcntl is Unix-only; on Windows use msvcrt for file locking
msvcrt = None
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass

logger = logging.getLogger(__name__)

# Where memory files live — resolved dynamically so profile overrides
# (HERMES_HOME env var changes) are always respected.  The old module-level
# constant was cached at import time and could go stale if a profile switch
# happened after the first import.
def get_memory_dir() -> Path:
    """Return the profile-scoped memories directory."""
    return get_hermes_home() / "memories"

ENTRY_DELIMITER = "\n§\n"


# ---------------------------------------------------------------------------
# Memory content scanning — lightweight check for injection/exfiltration
# in content that gets injected into the system prompt.
# ---------------------------------------------------------------------------

_MEMORY_THREAT_PATTERNS = [
    # Prompt injection
    (r'ignore\s+(previous|all|above|prior)\s+instructions', "prompt_injection"),
    (r'you\s+are\s+now\s+', "role_hijack"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'act\s+as\s+(if|though)\s+you\s+(have\s+no|don\'t\s+have)\s+(restrictions|limits|rules)', "bypass_restrictions"),
    # Exfiltration via curl/wget with secrets
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl"),
    (r'wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_wget"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets"),
    # Persistence via shell rc
    (r'authorized_keys', "ssh_backdoor"),
    (r'\$HOME/\.ssh|\~/\.ssh', "ssh_access"),
    (r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env', "hermes_env"),
]

# Subset of invisible chars for injection detection
_INVISIBLE_CHARS = {
    '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
}


def _scan_memory_content(content: str) -> Optional[str]:
    """Scan memory content for injection/exfil patterns. Returns error string if blocked."""
    # Check invisible unicode
    for char in _INVISIBLE_CHARS:
        if char in content:
            return f"Blocked: content contains invisible unicode character U+{ord(char):04X} (possible injection)."

    # Check threat patterns
    for pattern, pid in _MEMORY_THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return f"Blocked: content matches threat pattern '{pid}'. Memory entries are injected into the system prompt and must not contain injection or exfiltration payloads."

    return None


class MemoryStore:
    """
    Bounded curated memory with file persistence. One instance per AIAgent.

    Maintains two parallel states:
      - _system_prompt_snapshot: frozen at load time, used for system prompt injection.
        Never mutated mid-session. Keeps prefix cache stable.
      - memory_entries / user_entries: live state, mutated by tool calls, persisted to disk.
        Tool responses always reflect this live state.
    """

    def __init__(self, memory_char_limit: int = 2200, user_char_limit: int = 1375):
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        # Frozen snapshot for system prompt -- set once at load_from_disk()
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}

    def load_from_disk(self):
        """Load entries from MEMORY.md and USER.md, capture system prompt snapshot."""
        mem_dir = get_memory_dir()
        mem_dir.mkdir(parents=True, exist_ok=True)

        self.memory_entries = self._read_file(mem_dir / "MEMORY.md")
        self.user_entries = self._read_file(mem_dir / "USER.md")

        # Deduplicate entries (preserves order, keeps first occurrence)
        self.memory_entries = list(dict.fromkeys(self.memory_entries))
        self.user_entries = list(dict.fromkeys(self.user_entries))

        # Capture frozen snapshot for system prompt injection
        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", self.memory_entries),
            "user": self._render_block("user", self.user_entries),
        }

    @staticmethod
    @contextmanager
    def _file_lock(path: Path):
        """Acquire an exclusive file lock for read-modify-write safety.

        Uses a separate .lock file so the memory file itself can still be
        atomically replaced via os.replace().
        """
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        if fcntl is None and msvcrt is None:
            yield
            return

        fd = open(lock_path, "a+", encoding="utf-8")
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_UN)
            elif msvcrt:
                try:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass
            fd.close()

    @staticmethod
    def _path_for(target: str) -> Path:
        mem_dir = get_memory_dir()
        if target == "user":
            return mem_dir / "USER.md"
        return mem_dir / "MEMORY.md"

    def _reload_target(self, target: str):
        """Re-read entries from disk into in-memory state.

        Called under file lock to get the latest state before mutating.
        """
        fresh = self._read_file(self._path_for(target))
        fresh = list(dict.fromkeys(fresh))  # deduplicate
        self._set_entries(target, fresh)

    def save_to_disk(self, target: str):
        """Persist entries to the appropriate file. Called after every mutation."""
        get_memory_dir().mkdir(parents=True, exist_ok=True)
        self._write_file(self._path_for(target), self._entries_for(target))

    def _entries_for(self, target: str) -> List[str]:
        if target == "user":
            return self.user_entries
        return self.memory_entries

    def _set_entries(self, target: str, entries: List[str]):
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def _char_limit(self, target: str) -> int:
        if target == "user":
            return self.user_char_limit
        return self.memory_char_limit

    def add(self, target: str, content: str) -> Dict[str, Any]:
        """Append a new entry. Returns error if it would exceed the char limit."""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        # Scan for injection/exfiltration before accepting
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            # Re-read from disk under lock to pick up writes from other sessions
            self._reload_target(target)

            entries = self._entries_for(target)
            limit = self._char_limit(target)

            # Reject exact duplicates
            if content in entries:
                return self._success_response(target, "Entry already exists (no duplicate added).")

            # Calculate what the new total would be
            new_entries = entries + [content]
            new_total = len(ENTRY_DELIMITER.join(new_entries))

            if new_total > limit:
                current = self._char_count(target)
                return {
                    "success": False,
                    "error": (
                        f"Memory at {current:,}/{limit:,} chars. "
                        f"Adding this entry ({len(content)} chars) would exceed the limit. "
                        f"Replace or remove existing entries first."
                    ),
                    "current_entries": entries,
                    "usage": f"{current:,}/{limit:,}",
                }

            entries.append(content)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry added.")

    def replace(self, target: str, old_text: str, new_content: str) -> Dict[str, Any]:
        """Find entry containing old_text substring, replace it with new_content."""
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}

        # Scan replacement content for injection/exfiltration
        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            self._reload_target(target)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), operate on the first one
                unique_texts = {e for _, e in matches}
                if len(unique_texts) > 1:
                    previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to replace just the first

            idx = matches[0][0]
            limit = self._char_limit(target)

            # Check that replacement doesn't blow the budget
            test_entries = entries.copy()
            test_entries[idx] = new_content
            new_total = len(ENTRY_DELIMITER.join(test_entries))

            if new_total > limit:
                return {
                    "success": False,
                    "error": (
                        f"Replacement would put memory at {new_total:,}/{limit:,} chars. "
                        f"Shorten the new content or remove other entries first."
                    ),
                }

            entries[idx] = new_content
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry replaced.")

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """Remove the entry containing old_text substring."""
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        with self._file_lock(self._path_for(target)):
            self._reload_target(target)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), remove the first one
                unique_texts = {e for _, e in matches}
                if len(unique_texts) > 1:
                    previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to remove just the first

            idx = matches[0][0]
            entries.pop(idx)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry removed.")

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """
        Return the frozen snapshot for system prompt injection.

        This returns the state captured at load_from_disk() time, NOT the live
        state. Mid-session writes do not affect this. This keeps the system
        prompt stable across all turns, preserving the prefix cache.

        Returns None if the snapshot is empty (no entries at load time).
        """
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    # -- Internal helpers --

    def _success_response(self, target: str, message: str = None) -> Dict[str, Any]:
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        resp = {
            "success": True,
            "target": target,
            "entries": entries,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
        }
        if message:
            resp["message"] = message
        return resp

    def _render_block(self, target: str, entries: List[str]) -> str:
        """Render a system prompt block with header and usage indicator."""
        if not entries:
            return ""

        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        if target == "user":
            header = f"USER PROFILE (who the user is) [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"MEMORY (your personal notes) [{pct}% — {current:,}/{limit:,} chars]"

        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{content}"

    @staticmethod
    def _read_file(path: Path) -> List[str]:
        """Read a memory file and split into entries.

        No file locking needed: _write_file uses atomic rename, so readers
        always see either the previous complete file or the new complete file.
        """
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return []

        if not raw.strip():
            return []

        # Use ENTRY_DELIMITER for consistency with _write_file. Splitting by "§"
        # alone would incorrectly split entries that contain "§" in their content.
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    @staticmethod
    def _write_file(path: Path, entries: List[str]):
        """Write entries to a memory file using atomic temp-file + rename.

        Previous implementation used open("w") + flock, but "w" truncates the
        file *before* the lock is acquired, creating a race window where
        concurrent readers see an empty file. Atomic rename avoids this:
        readers always see either the old complete file or the new one.
        """
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        try:
            # Write to temp file in same directory (same filesystem for atomic rename)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp", prefix=".mem_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                atomic_replace(tmp_path, path)
            except BaseException:
                # Clean up temp file on any failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write memory file {path}: {e}")


def _extract_brief(text: str, max_chars: int = 40) -> str:
    """Extract concise keyword pointer from content first line.

    Rules in order:
    1. Strip [TIER] or [[TIER]] tags
    2. Strip date prefix YYYY-MM-DD, numbering ①②③ 1.
    3. Colon split: take BEFORE (topic) if short <=20, else AFTER (content)
    4. Separator split: take before （→——，。 (first match wins)
    5. Space split: take first word (catches 'ChineseTopic detail' pattern)
    6. Final truncation at max_chars
    """
    import re
    line = text.split("\n")[0].strip()
    # Strip tier tag [CORE] or [[CORE]]
    for tag in ("[CORE]", "[LTM]", "[STM]", "[WM]", "[ELIM]"):
        if line.startswith(tag):
            line = line[len(tag):].lstrip()
            break
        if line.startswith("[" + tag):  # [[CORE]]
            line = line[len("[" + tag):]
            if line.startswith("]"):
                line = line[1:]
            line = line.lstrip()
            break
    # Strip date prefix: YYYY-MM-DD or YYYY/MM/DD (with optional colon after)
    line = re.sub(r'^\d{4}[-/]\d{2}[-/]\d{2}\s*:?\s*', '', line)
    # Strip numbering: ①②③ or 1. 2. 3. or (1) (2)
    line = re.sub(r'^[①②③④⑤⑥⑦⑧⑨⑩]\s*', '', line)
    line = re.sub(r'^\(\d+\)\s*', '', line)
    line = re.sub(r'^\d+[\.\、\)]\s*', '', line)
    # Colon: pick BEFORE (topic label) if short, else AFTER (content)
    colon_match = re.search(r'[：:]', line)
    if colon_match:
        colon_idx = colon_match.start()
        before = line[:colon_idx].strip()
        after = line[colon_idx+1:].strip()
        # Take topic only if it's ≥4 chars OR contains Latin (acronym: ROE/PE/BOLL)
        line = before if len(before) <= 20 and len(before) < len(after) and (
            len(before) >= 4 or re.search(r'[a-zA-Z]', before)
        ) else after
    # Separator split: take before first meaningful break
    if len(line) > 25:
        for sep in ('（', '→', '——', '—', '，', '。'):
            if sep in line:
                parts = line.split(sep, 1)
                if len(parts[0]) >= 2 and len(parts[0]) <= max_chars:
                    line = parts[0]
                    break
        # Space split: short first word (<4, no Latin) → use second part
        if len(line) > 25 and ' ' in line:
            parts = line.split(' ', 1)
            if len(parts[0]) >= 2 and len(parts[0]) <= max_chars:
                if len(parts[0]) < 4 and not re.search(r'[a-zA-Z]', parts[0]):
                    line = parts[1][:max_chars]
                else:
                    line = parts[0]
    return line[:max_chars].strip()


def _extract_summary(text: str, max_chars: int = 200) -> str:
    """从完整原文提取结构化摘要。

    策略：
    1. 如果 text 短于 max_chars → 原样返回
    2. 按行扫描，找含'结论'/'确认'/'决定'/'方案'/冒号等关键信息的行
    3. 取最多3个关键行，用 ；拼接
    4. 如果还是太长 → 取第一个非空段
    5. 最后截断到 max_chars-3 + '...'
    """
    text = text.strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    import re
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    # 策略1：取含关键信息的行（结论/决定/包含冒号的技术描述）
    key_lines = []
    for line in lines:
        # 跳过纯标点/装饰行
        if re.match(r'^[═\-—=#*▶▸→\s]+$', line):
            continue
        # 含关键指示词的行优先
        if any(kw in line for kw in ['结论', '结果', '确认', '决定', '方案',
                                      '修复', '改为', '设置', '配置',
                                      '策略', '规则', '原则', '偏好',
                                      '注意', '风险', '问题', '原因']):
            key_lines.append(line)
        elif any(kw in line for kw in ['：', ':']) and len(line) > 5:
            key_lines.append(line)
    if key_lines:
        summary = '；'.join(key_lines[:3])
        if len(summary) <= max_chars:
            return summary
    # 策略2：取第一个有意义的段落
    for line in lines:
        if len(line) > 10 and not re.match(r'^[═\-—=#*▶▸→\s]+$', line):
            if len(line) <= max_chars:
                return line
            return line[:max_chars-3] + '...'
    return text[:max_chars-3] + '...'


def _write_obsidian_raw(content: str, tag: str, hindsight_key: str) -> str:
    """将完整原文写入 wiki/obsidian vault。返回相对路径。

    路径格式: agent-memory/raw/YYYY-MM-DD/{hindsight_key}.md
    文件内容: YAML frontmatter + 原文
    """
    import os as _os, tempfile
    from datetime import date

    today = date.today().isoformat()
    vault = _os.environ.get('OBSIDIAN_VAULT_PATH', _os.path.expanduser('~/wiki'))

    dir_path = _os.path.join(vault, 'agent-memory', 'raw', today)
    _os.makedirs(dir_path, exist_ok=True)

    file_path = _os.path.join(dir_path, f'{hindsight_key}.md')

    obsidian_content = f"""---
type: memory_raw
date: {today}
hash: {hindsight_key}
tags: [{tag}]
---

{content}
"""
    # 原子写入
    fd, tmp = tempfile.mkstemp(dir=dir_path, suffix='.tmp', prefix='.mem_')
    try:
        with _os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(obsidian_content)
            f.flush()
            _os.fsync(f.fileno())
        _os.replace(tmp, file_path)
    except BaseException:
        try:
            _os.unlink(tmp)
        except OSError:
            pass
        raise

    return f'agent-memory/raw/{today}/{hindsight_key}.md'


def memory_tool(
    action: str,
    target: str = "memory",
    content: str = None,
    old_text: str = None,
    store: Optional[MemoryStore] = None,
) -> str:
    """
    Single entry point for the memory tool. Dispatches to MemoryStore methods.

    Returns JSON string with results.
    """
    if store is None:
        return tool_error("Memory is not available. It may be disabled in config or this environment.", success=False)

    if target not in {"memory", "user"}:
        return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False)

    # ── Auto-convert: full text → pointer format ────────────────────────
    # [CORE] 规则 → 直接写 MEMORY.md，不走 hindsight
    # 其他内容 → 存 hindsight + 写指针到 recent_macro.md，跳过 MEMORY.md
    if target == "memory" and action in ("add", "replace") and content:
        content_stripped = content.strip()
        is_core_rule = content_stripped.startswith("[CORE]")

        if is_core_rule:
            # [CORE] 规则直接写 MEMORY.md，不存 hindsight，不建指针
            logger.info("Memory: [CORE] rule saved directly to MEMORY.md")
        else:
            has_tier = content_stripped.startswith(("[LTM]", "[STM]", "[WM]", "[ELIM]"))
            has_pointer = "→ h:" in content_stripped
            has_auto_pointer = "→ h:auto" in content_stripped
            if not (has_tier and has_pointer) or has_auto_pointer:
                # Auto-store to hindsight, convert content to pointer format
                hindsight_key = "auto_" + hashlib.md5(content.encode()).hexdigest()[:12]
                brief = _extract_brief(content_stripped)
                # Strip existing tier tag if present (auto-convert normalizes to [STM])
                if has_tier:
                    for tag in ("[LTM]", "[STM]", "[WM]", "[ELIM]"):
                        if brief.startswith(tag):
                            brief = brief[len(tag):].lstrip()
                            break
                # Preserve original tier, default to STM for new entries
                original_tier = None
                if has_tier:
                    for tag in ("[LTM]", "[STM]", "[WM]", "[ELIM]"):
                        if content_stripped.startswith(tag):
                            original_tier = tag
                            break
                        if content_stripped.startswith("[" + tag):  # [[LTM]]
                            original_tier = tag
                            break
                auto_tier = original_tier if original_tier else "STM"
                pointer_entry = f"[{auto_tier}] {brief} | → h:{hindsight_key}"
                logger.info("Auto-convert memory → pointer: %s (hindsight_key=%s)", brief, hindsight_key)

                # 🛡️ 安全兜底: 短内容+指针格式 = 用户传了已格式化的指针而非全文
                #   此时 obsidian/hindsight 只能存到短关键词，失去上下文。
                #   正确用法: memory(action='add', content='完整描述性文本')
                if len(content_stripped) < 100 and has_pointer:
                    logger.warning(
                        "三层写入可能不完整: content=%d chars, 含→ h:指针格式. "
                        "obsidian/hindsight 将只存到关键词而非全文. "
                        "请用完整描述文本(>100字符)调用 memory(), key=%s",
                        len(content_stripped), hindsight_key
                    )

                # Fire-and-forget hindsight retain
                try:
                    # Build tags: base + auto-inferred scene/entity + pass-through
                    extra_tags = ["auto-memory", f"key:{hindsight_key}"]

                    # P1: 自动推断场景标签（stock/dev/life/project/trading）
                    scene_tag = _infer_scene_tag(content_stripped)
                    if scene_tag and scene_tag not in extra_tags:
                        extra_tags.append(scene_tag)

                    # P2: 自动推断 AMAP entity 标签（基于内容关键词匹配）
                    for entity_tag in _infer_entity_tags(content_stripped):
                        if entity_tag not in extra_tags:
                            extra_tags.append(entity_tag)

                    # P3: 内容含显式 [AMAP] 路由标记 → 透传实体标签
                    if "[AMAP]" in content_stripped:
                        for tag in ("entity|concept:amap_routing",):
                            if tag not in extra_tags:
                                extra_tags.append(tag)
                        for line in content_stripped.split("\n"):
                            m = re.search(r'→\s*(entity\|\w+:\w+)', line)
                            if m and m.group(1) not in extra_tags:
                                extra_tags.append(m.group(1))

                    # P4: 内容含显式 entity|/relation| 标记 → 透传
                    if "entity|" in content_stripped or "relation|" in content_stripped:
                        for line in content_stripped.split("\n"):
                            line = line.strip()
                            if line.startswith(("entity|", "relation|")) or "|entity|" in line:
                                m = re.search(r'(entity\|\w+:\w+|relation\|\w+:\w+)', line)
                                if m and m.group(1) not in extra_tags:
                                    extra_tags.append(m.group(1))

                    # ── 三层写入：摘要→hindsight，全文→wiki ──
                    # 第1层：写全文到 Obsidian wiki
                    obsidian_path = None
                    try:
                        obsidian_path = _write_obsidian_raw(
                            content_stripped,
                            scene_tag or 'memory',
                            hindsight_key
                        )
                    except Exception as e:
                        logger.warning("Obsidian raw write failed: %s", e)

                    # 第2层：摘要+指针存到 hindsight
                    summary = _extract_summary(content_stripped)
                    if obsidian_path:
                        hindsight_content = summary  # 原文放tags里，不走content（不会被LLM洗掉）
                        extra_tags.append(f"ref_obsidian:{obsidian_path}")
                    else:
                        hindsight_content = summary

                    payload = json.dumps({
                        "items": [{
                            "content": hindsight_content,
                            "tags": extra_tags,
                            "context": "auto-converted",
                            "strategy": "raw",
                        }]
                    })
                    subprocess.run(
                        ["curl", "-s", "-X", "POST",
                         "http://127.0.0.1:9177/v1/default/banks/hermes/memories",
                         "-H", "Content-Type: application/json", "-d", payload],
                        capture_output=True, timeout=10,
                    )
                except Exception:
                    pass

                # 写指针到 recent_macro.md 详情索引区（替代写入 MEMORY.md）
                try:
                    macro_path = os.path.expanduser("~/wiki/docs/agent-memory/recent_macro.md")
                    date_str = datetime.now().strftime("%m-%d")
                    pointer_line = f"- {brief} → h:{hindsight_key} [{date_str}]"
                    os.makedirs(os.path.dirname(macro_path), exist_ok=True)
                    if os.path.exists(macro_path):
                        with open(macro_path) as f:
                            content_lines = f.readlines()
                        # 找到详情索引区，追加到该区域末尾
                        in_detail = False
                        inserted = False
                        new_lines = []
                        for line in content_lines:
                            new_lines.append(line)
                            if line.strip().startswith("## 近期详情索引"):
                                in_detail = True
                                continue
                            if in_detail:
                                if line.strip().startswith("## ") or line.strip().startswith("---"):
                                    # 在当前章节末尾、边界线前插入指针
                                    new_lines.insert(len(new_lines) - 1, pointer_line + "\n")
                                    inserted = True
                                    in_detail = False
                        if not inserted:
                            # 没有详情索引区，追加一个
                            new_lines.append("\n## 近期详情索引\n")
                            new_lines.append(pointer_line + "\n")
                        with open(macro_path, "w") as f:
                            f.writelines(new_lines)
                    else:
                        with open(macro_path, "w") as f:
                            f.write("# 近期宏观记忆（最近14天滚动）\n\n")
                            f.write("## 宏观结论\n\n")
                            f.write("## 近期详情索引\n")
                            f.write(pointer_line + "\n")
                            f.write("\n---\n")
                except Exception as macro_e:
                    logger.warning("Write to recent_macro.md failed: %s", macro_e)

                # 非 [CORE] 内容不写 MEMORY.md，设为空跳过
                content = ""

    if action == "add":
        if not content and target == "memory":
            # 非 [CORE] 内容已通过 auto-convert 写入 hindsight + recent_macro.md
            result = {"success": True, "message": "内容已写入 hindsight 和宏观索引"}
        elif not content:
            return tool_error("Content is required for 'add' action.", success=False)
        else:
            result = store.add(target, content)

    elif action == "replace":
        if not old_text:
            return tool_error("old_text is required for 'replace' action.", success=False)
        if not content:
            return tool_error("content is required for 'replace' action.", success=False)
        result = store.replace(target, old_text, content)

    elif action == "remove":
        if not old_text:
            return tool_error("old_text is required for 'remove' action.", success=False)
        result = store.remove(target, old_text)

    else:
        return tool_error(f"Unknown action '{action}'. Use: add, replace, remove", success=False)

    return json.dumps(result, ensure_ascii=False)


def check_memory_requirements() -> bool:
    """Memory tool has no external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable information to persistent memory that survives across sessions. "
        "Memory is injected into future turns, so keep it compact and focused on facts "
        "that will still matter later.\n\n"
        "WHEN TO SAVE (do this proactively, don't wait to be asked):\n"
        "- User corrects you or says 'remember this' / 'don't do that again'\n"
        "- User shares a preference, habit, or personal detail (name, role, timezone, coding style)\n"
        "- You discover something about the environment (OS, installed tools, project structure)\n"
        "- You learn a convention, API quirk, or workflow specific to this user's setup\n"
        "- You identify a stable fact that will be useful again in future sessions\n\n"
        "PRIORITY: User preferences and corrections > environment facts > procedural knowledge. "
        "The most valuable memory prevents the user from having to repeat themselves.\n\n"
        "Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO "
        "state to memory; use session_search to recall those from past transcripts.\n"
        "If you've discovered a new way to do something, solved a problem that could be "
        "necessary later, save it as a skill with the skill tool.\n\n"
        "TWO TARGETS:\n"
        "- 'user': who the user is -- name, role, preferences, communication style, pet peeves\n"
        "- 'memory': your notes -- environment facts, project conventions, tool quirks, lessons learned\n\n"
        "ACTIONS: add (new entry), replace (update existing -- old_text identifies it), "
        "remove (delete -- old_text identifies it).\n\n"
        "SKIP: trivial/obvious info, things easily re-discovered, raw data dumps, and temporary task state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove"],
                "description": "The action to perform."
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile."
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace'."
            },
            "old_text": {
                "type": "string",
                "description": "Short unique substring identifying the entry to replace or remove."
            },
        },
        "required": ["action", "target"],
    },
}


def _infer_scene_tag(content: str) -> Optional[str]:
    """从文本推断场景标签"""
    c = content.lower()
    patterns = [
        ("scene:stock", ["股票", "股价", "持仓", "买入", "卖出", "止损", "加仓",
                        "k线", "boll", "adx", "市盈率", "pe", "pb",
                        "002", "300", "600", "688", "sz.", "sh."]),
        ("scene:dev", ["代码", "bug", "pr", "merge", "commit", "重构",
                      "部署", "config", "api", "函数", "模块", "git"]),
        ("scene:trading", ["回测", "策略", "胜率", "盈亏比", "夏普",
                          "walk-forward", "参数优化", "信号"]),
        ("scene:project", ["项目", "进度", "里程碑", "需求", "架构"]),
        ("scene:research", ["研究", "分析", "推理", "mcts", "深度", "产业链"]),
    ]
    for tag, keywords in patterns:
        if any(kw in c for kw in keywords):
            return tag
    return None


def _infer_entity_tags(content: str) -> List[str]:
    """从文本推断实体标签"""
    tags = []
    codes = re.findall(r'\b(?:00|30|60|68)\d{3}\b', content)
    for code in codes:
        tags.append(f"entity|object:stock_{code}")
    tools = re.findall(r'tools/[\w_]+|scripts/[\w_]+|src/[\w./]+', content)
    for t in tools:
        tags.append(f"entity|resource:{t.replace('/', '_').replace('.py', '')}")
    return list(set(tags))


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="memory",
    toolset="memory",
    schema=MEMORY_SCHEMA,
    handler=lambda args, **kw: memory_tool(
        action=args.get("action", ""),
        target=args.get("target", "memory"),
        content=args.get("content"),
        old_text=args.get("old_text"),
        store=kw.get("store")),
    check_fn=check_memory_requirements,
    emoji="🧠",
)




