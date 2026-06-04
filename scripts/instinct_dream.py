#!/usr/bin/env python3
"""
做梦精炼 — 每日触发（cron 03:00）

每日流程：
  1-6. 本能精炼（分析观察 → 更新 YAML → 裁剪）
  7.    MEMORY.md 健康检查 + 周日自动整理

非关键路径：任何步骤失败都只记录日志，不阻止后续步骤。
"""
import sys
import os
import re
from pathlib import Path
from datetime import datetime, timezone

# ── 路径 ──────────────────────────────────────────────
_HERMES_AGENT = Path(__file__).resolve().parent.parent
if str(_HERMES_AGENT) not in sys.path:
    sys.path.insert(0, str(_HERMES_AGENT))

_HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
_MEMORY_FILE = _HERMES_HOME / "memories" / "MEMORY.md"

# MEMORY 检查阈值
_MAX_CORE_CHARS = 2000     # CORE 区最大字符数
_MAX_CORE_ENTRIES = 25     # CORE 区最大条目数
_MAX_ARCHIVE_CHARS = 3000  # ARCHIVE 区最大字符数

from agent.instincts import (
    analyze_observations,
    save_instincts,
    _maybe_trim_observations,
    _load_observations,
    _OBSERVATIONS_FILE,
    _MAX_OBSERVATIONS,
    get_high_confidence_instincts,
)


# ── MEMORY 精炼 ──────────────────────────────────────

_MEMORY_HEADER = "══════════════════════════════════════════════\nMEMORY (your personal notes)\n══════════════════════════════════════════════"

_CORE_HEADER = "# ── CORE（硬性规范，持续注入） ──────────────"
_ARCHIVE_HEADER = "# ── ARCHIVE（历史记录，不注入） ────────────"


def _parse_memory_sections(text: str) -> dict:
    """将 MEMORY.md 解析为分层结构。"""
    result = {
        "preamble": "",
        "core": [],
        "archive": [],
        "postamble": "",
    }
    current_section = "preamble"
    for line in text.split("\n"):
        stripped = line.strip()
        if _CORE_HEADER in stripped:
            current_section = "core"
            result["preamble"] += line + "\n"
        elif _ARCHIVE_HEADER in stripped:
            current_section = "archive"
            result["core"] = result["core"][:-1]  # 移除 header 行，单独存
            result["core_header"] = line
        elif current_section == "preamble":
            result["preamble"] += line + "\n"
        elif current_section == "core":
            if stripped.startswith("[") and stripped != "":
                result["core"].append(line)
        elif current_section == "archive":
            if stripped.startswith("[") and stripped != "":
                result["archive"].append(line)
            else:
                result["postamble"] += line + "\n"

    return result


def _check_memory_health() -> dict:
    """检查 MEMORY.md 健康度，返回统计数据。"""
    result = {
        "core_count": 0,
        "core_chars": 0,
        "archive_count": 0,
        "archive_chars": 0,
        "core_over_threshold": False,
        "archive_over_threshold": False,
        "stray_archive": [],  # 在 CORE 区但标记 [ARCHIVE] 的条目
        "warnings": [],
        "sunday_refine": False,
    }

    if not _MEMORY_FILE.exists():
        result["warnings"].append("MEMORY.md not found")
        return result

    text = _MEMORY_FILE.read_text(encoding="utf-8")
    sections = _parse_memory_sections(text)

    # CORE 统计
    for entry in sections["core"]:
        stripped = entry.strip()
        if not stripped:
            continue
        result["core_count"] += 1
        result["core_chars"] += len(stripped)
        # 检测 CORE 区中的 [ARCHIVE] 标记
        if stripped.startswith("[ARCHIVE]"):
            result["stray_archive"].append(stripped)

    # ARCHIVE 统计
    for entry in sections["archive"]:
        stripped = entry.strip()
        if not stripped:
            continue
        result["archive_count"] += 1
        result["archive_chars"] += len(stripped)

    # 阈值检查
    if result["core_chars"] > _MAX_CORE_CHARS:
        result["core_over_threshold"] = True
        result["warnings"].append(
            f"CORE 区 {result['core_chars']} chars > {_MAX_CORE_CHARS} 阈值"
        )
    if result["core_count"] > _MAX_CORE_ENTRIES:
        result["core_over_threshold"] = True
        result["warnings"].append(
            f"CORE 区 {result['core_count']} 条 > {_MAX_CORE_ENTRIES} 阈值"
        )
    if result["archive_chars"] > _MAX_ARCHIVE_CHARS:
        result["archive_over_threshold"] = True
        result["warnings"].append(
            f"ARCHIVE 区 {result['archive_chars']} chars > {_MAX_ARCHIVE_CHARS} 阈值"
        )

    return result


def _refine_memory() -> dict:
    """MEMORY.md 自动精炼（周日执行）。

    机械操作（无需 LLM）：
      1. 将 CORE 区中标记 [ARCHIVE] 的条目移到 ARCHIVE 区
      2. 去重完全相同的条目
      3. 移除多余空行，保持格式整洁
      4. 确认 CORE 领先 ARCHIVE 在前
    """
    result = {"moved_to_archive": 0, "dedup_removed": 0, "steps": []}

    if not _MEMORY_FILE.exists():
        result["steps"].append("MEMORY.md not found, skip")
        return result

    text = _MEMORY_FILE.read_text(encoding="utf-8")
    sections = _parse_memory_sections(text)

    # Step 1: 把 CORE 区中的 [ARCHIVE] 条目移下去
    clean_core = []
    moved = []
    for entry in sections["core"]:
        stripped = entry.strip()
        if stripped.startswith("[ARCHIVE]"):
            moved.append(entry)
        else:
            clean_core.append(entry)
    result["moved_to_archive"] = len(moved)
    if moved:
        result["steps"].append(f"moved {len(moved)} stray ARCHIVE items to archive")

    # Step 2: 去重（完全相同的行）
    seen = set()
    unique_core = []
    for entry in clean_core:
        stripped = entry.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            unique_core.append(entry)
        elif stripped in seen:
            result["dedup_removed"] += 1
    if result["dedup_removed"] > 0:
        result["steps"].append(f"dedup {result['dedup_removed']} core entries")

    # 合并 archive（把被移下来的加进去）
    archive_entries = sections.get("archive", []) + moved
    seen_archive = set()
    unique_archive = []
    for entry in archive_entries:
        stripped = entry.strip()
        if stripped and stripped not in seen_archive:
            seen_archive.add(stripped)
            unique_archive.append(entry)

    # Step 3: 写回文件
    parts = [
        _MEMORY_HEADER,
        "",
        _CORE_HEADER,
        "",
    ]
    parts.extend(unique_core)
    parts.append("")
    parts.append(_ARCHIVE_HEADER)
    parts.append("")
    parts.extend(unique_archive)
    if sections.get("postamble", "").strip():
        parts.append("")
        parts.append(sections["postamble"].strip())

    _MEMORY_FILE.write_text("\n".join(parts), encoding="utf-8")
    result["steps"].append(f"rewrote MEMORY.md ({len(unique_core)} core, {len(unique_archive)} archive)")

    return result


# ── 主做梦流程 ──────────────────────────────────────

def dream() -> dict:
    """单次做梦精炼回合。返回状态报告。"""
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "observations_before": 0,
        "observations_after": 0,
        "instincts_before": 0,
        "instincts_generated": 0,
        "instincts_after": 0,
        "high_confidence_90p": 0,
        "observations_file_bytes": 0,
        "memory_core_count": 0,
        "memory_core_chars": 0,
        "memory_archive_count": 0,
        "memory_warnings": [],
        "memory_refine": None,
        "steps": [],
    }

    # ── Steps 1-6: 本能精炼（与原来一致） ──

    # Step 1: 观察数据量
    try:
        obs = _load_observations()
        report["observations_before"] = len(obs)
        report["observations_file_bytes"] = (
            _OBSERVATIONS_FILE.stat().st_size if _OBSERVATIONS_FILE.exists() else 0
        )
        report["steps"].append("load_observations: OK")
    except Exception as e:
        report["steps"].append(f"load_observations: FAILED ({e})")
        return report

    # Step 2: 本能计数（精炼前）
    try:
        before = get_high_confidence_instincts(threshold=0.0)
        report["instincts_before"] = len(before)
        report["high_confidence_90p"] = sum(
            1 for i in before if i.get("confidence", 0) >= 0.9
        )
        report["steps"].append("count_before: OK")
    except Exception as e:
        report["steps"].append(f"count_before: FAILED ({e})")

    # Step 3: 分析观察 → 生成新本能
    try:
        candidates = analyze_observations()
        report["instincts_generated"] = len(candidates)
        report["steps"].append(f"analyze: OK ({len(candidates)} candidates)")
    except Exception as e:
        report["steps"].append(f"analyze: FAILED ({e})")
        candidates = []

    # Step 4: 保存到 YAML
    if candidates:
        try:
            save_instincts(candidates)
            report["steps"].append("save_instincts: OK")
        except Exception as e:
            report["steps"].append(f"save_instincts: FAILED ({e})")

    # Step 5: 本能计数（精炼后）
    try:
        after = get_high_confidence_instincts(threshold=0.0)
        report["instincts_after"] = len(after)
        report["high_confidence_90p"] = sum(
            1 for i in after if i.get("confidence", 0) >= 0.9
        )
        report["steps"].append("count_after: OK")
    except Exception as e:
        report["steps"].append(f"count_after: FAILED ({e})")

    # Step 6: 裁剪观察文件
    try:
        _maybe_trim_observations()
        report["observations_after"] = len(_load_observations())
        report["steps"].append("trim_observations: OK")
    except Exception as e:
        report["steps"].append(f"trim_observations: FAILED ({e})")

    # ── Step 7: MEMORY.md 健康检查 + 精炼 ──
    try:
        health = _check_memory_health()
        report["memory_core_count"] = health["core_count"]
        report["memory_core_chars"] = health["core_chars"]
        report["memory_archive_count"] = health["archive_count"]
        report["memory_warnings"] = health["warnings"]

        # 周日自动执行精炼
        today = datetime.now(timezone.utc)
        if today.weekday() == 6:  # Sunday
            refine_result = _refine_memory()
            report["memory_refine"] = refine_result
            report["steps"].append(
                f"memory_refine: OK (moved {refine_result['moved_to_archive']}, "
                f"dedup {refine_result['dedup_removed']})"
            )
        else:
            report["steps"].append("memory_check: OK")
    except Exception as e:
        report["steps"].append(f"memory_check: FAILED ({e})")

    return report


def format_report(report: dict) -> str:
    """格式化为可读报告。"""
    lines = [
        f"🧠 做梦精炼报告 — {report['timestamp'][:19]}",
        f"",
    ]

    # 本能部分
    lines.append(f"  📊 本能: {report['instincts_before']} → {report['instincts_after']} 条 "
                 f"(生成 {report['instincts_generated']} 候选, ≥90%: {report['high_confidence_90p']} 条)")
    lines.append(f"  📂 观察: {report['observations_before']:,} 行 → {report['observations_after']:,} 行 "
                 f"({report['observations_file_bytes']/1024/1024:.1f} MB)")

    # MEMORY 部分
    mem_line = f"  📝 MEMORY: {report['memory_core_count']} 条 CORE ({report['memory_core_chars']} chars), {report['memory_archive_count']} 条 ARCHIVE"
    if report["memory_warnings"]:
        mem_line += " ⚠️"
    else:
        mem_line += " ✅"
    lines.append(mem_line)

    if report["memory_warnings"]:
        for w in report["memory_warnings"]:
            lines.append(f"    ⚠️  {w}")

    if report["memory_refine"]:
        lines.append(f"  🧹 周日精炼: moved {report['memory_refine']['moved_to_archive']} stray, "
                     f"dedup {report['memory_refine']['dedup_removed']}")

    # 步骤
    lines.append(f"  🔄 步骤: {' | '.join(report['steps'])}")

    return "\n".join(lines)


if __name__ == "__main__":
    report = dream()
    print(format_report(report))

    # 非零退出码表示关键步骤失败
    failed = [s for s in report["steps"] if "FAILED" in s]
    if failed:
        print(f"\n⚠️  {len(failed)} 步骤失败:", file=sys.stderr)
        for s in failed:
            print(f"   {s}", file=sys.stderr)
        sys.exit(1)
