#!/usr/bin/env python3
"""
本能做梦精炼 — 每日触发（cron 03:00）

流程：
  1. 读取观察数据
  2. 分析模式 → 生成/更新 YAML 本能（consolidate）
  3. 裁剪观察文件（原始数据已精炼进 YAML，旧数据可安全移除）

非关键路径：任何步骤失败都只记录日志，不阻止后续步骤。
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone

# ── 路径 ──────────────────────────────────────────────
# 确保能找到 agent/instincts.py
_HERMES_AGENT = Path(__file__).resolve().parent.parent
if str(_HERMES_AGENT) not in sys.path:
    sys.path.insert(0, str(_HERMES_AGENT))

from agent.instincts import (
    analyze_observations,
    save_instincts,
    _maybe_trim_observations,
    _load_observations,
    _OBSERVATIONS_FILE,
    _INSTINCTS_FILE,
    _MAX_OBSERVATIONS,
    get_high_confidence_instincts,
)


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
        "steps": [],
    }

    # ── Step 1: 观察数据量 ──
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

    # ── Step 2: 本能计数（精炼前） ──
    try:
        before = get_high_confidence_instincts(threshold=0.0)
        report["instincts_before"] = len(before)
        report["high_confidence_90p"] = sum(
            1 for i in before if i.get("confidence", 0) >= 0.9
        )
        report["steps"].append("count_before: OK")
    except Exception as e:
        report["steps"].append(f"count_before: FAILED ({e})")

    # ── Step 3: 分析观察 → 生成新本能（精炼核心） ──
    try:
        candidates = analyze_observations()
        report["instincts_generated"] = len(candidates)
        report["steps"].append(f"analyze: OK ({len(candidates)} candidates)")
    except Exception as e:
        report["steps"].append(f"analyze: FAILED ({e})")
        candidates = []

    # ── Step 4: 保存到 YAML（合并到已有本能） ──
    if candidates:
        try:
            save_instincts(candidates)
            report["steps"].append("save_instincts: OK")
        except Exception as e:
            report["steps"].append(f"save_instincts: FAILED ({e})")

    # ── Step 5: 本能计数（精炼后） ──
    try:
        after = get_high_confidence_instincts(threshold=0.0)
        report["instincts_after"] = len(after)
        report["high_confidence_90p"] = sum(
            1 for i in after if i.get("confidence", 0) >= 0.9
        )
        report["steps"].append("count_after: OK")
    except Exception as e:
        report["steps"].append(f"count_after: FAILED ({e})")

    # ── Step 6: 裁剪观察文件（原始数据已精炼进 YAML） ──
    try:
        _maybe_trim_observations()
        report["observations_after"] = len(_load_observations())
        report["steps"].append("trim_observations: OK")
    except Exception as e:
        report["steps"].append(f"trim_observations: FAILED ({e})")

    return report


def format_report(report: dict) -> str:
    """格式化为可读报告。"""
    lines = [
        f"🧠 本能做梦精炼报告 — {report['timestamp'][:19]}",
        f"",
        f"  观察文件: {report['observations_before']:,} 行 → {report['observations_after']:,} 行 "
        f"({report['observations_file_bytes']/1024/1024:.1f} MB)",
        f"  本能: {report['instincts_before']} → {report['instincts_after']} 条 "
        f"(生成 {report['instincts_generated']} 候选, ≥90%: {report['high_confidence_90p']} 条)",
        f"  步骤: {' | '.join(report['steps'])}",
    ]
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
