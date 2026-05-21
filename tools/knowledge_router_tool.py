#!/usr/bin/env python3
"""
知识路由工具 — 统一知识检索（wiki + hindsight + graph 联想 + skill推荐）

自动调 ~/.hermes/scripts/knowledge_router.py 实现四层检索：
  1. wiki (Vector RAG) → 已 curated 的结构化知识
  2. hindsight (Episodic Memory) → 对话中自动沉淀的经验
  3. graph (联想记忆) → 实体图扩散，搜A联想到关联的B/C/D
  4. skill (技能推荐) → 根据查询匹配最相关的 skill

用法（在对话中直接调）：
  knowledge_router("卫星化学")           # 自动路由到对应层
  knowledge_router("卫星化学", source="all")  # 全层搜索+图扩散联想+skill推荐
  knowledge_router("卫星化学", source="graph") # 只做联想记忆扩散
  knowledge_router("持仓分析", source="skill") # 只做技能推荐
"""

import json
import logging
import os
import shlex
import subprocess
import sys

logger = logging.getLogger(__name__)

SCRIPT_PATH = os.path.expanduser("~/.hermes/scripts/knowledge_router.py")


def check_requirements() -> bool:
    """脚本存在即可用"""
    return os.path.exists(SCRIPT_PATH)


def knowledge_router(
    query: str,
    source: str = "auto",
    limit: int = 5,
) -> str:
    """
    统一知识检索：从 wiki + hindsight + graph + skill 四层检索与 query 相关的内容。
    类似于 session_search 但搜索的是永久知识库（wiki/经验记忆/联想图/skill库），而非对话记录。

    Args:
        query: 搜索关键词，如 "卫星化学"、"ADX指标"、"BOLL和ADX的关系"
        source: 搜索范围
            "auto"  (默认) → 自动分类查询类型后路由到对应层
            "all"           → 全层搜索（wiki + hindsight + graph图扩散联想 + skill推荐）
            "wiki"          → 只搜 wiki 知识库
            "hindsight"     → 只搜 hindsight 经验记忆
            "graph"         → 只做图扩散联想（搜A→关联到B/C/D）
            "skill"         → 只做技能推荐（匹配最相关的 skill）
        limit: 每层最多返回结果数

    Returns:
        JSON 字符串，含 results 列表，每条含 source/score/content/title 等字段
    """
    if not os.path.exists(SCRIPT_PATH):
        return json.dumps({
            "success": False,
            "error": f"知识路由脚本不存在: {SCRIPT_PATH}",
            "query": query,
            "source": source,
        }, ensure_ascii=False)

    cmd = [
        sys.executable, SCRIPT_PATH,
        query,
    ]

    if source == "skill":
        cmd.append("--skill")
    elif source == "all":
        cmd.append("--all")
    elif source == "graph":
        cmd.append("--graph")
    elif source != "auto":
        cmd.append(f"--source={source}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        if result.returncode != 0:
            return json.dumps({
                "success": False,
                "error": f"知识路由调用失败 (exit={result.returncode}): {result.stderr[:500]}",
                "query": query,
                "source": source,
            }, ensure_ascii=False)

        raw = result.stdout.strip()
        if not raw:
            return json.dumps({
                "success": True,
                "query": query,
                "source": source,
                "results": [],
                "summary": "无匹配结果",
            }, ensure_ascii=False)

        # 解析结构化输出
        lines = raw.split("\n")
        results = []
        current = {}
        current_source = ""
        for line in lines:
            if line.startswith("  📖 [wiki]"):
                if current:
                    results.append(current)
                current = {"source": "wiki", "score": _parse_score(line)}
                current_source = "wiki"
            elif line.startswith("  🧠 [hindsight]"):
                if current:
                    results.append(current)
                current = {"source": "hindsight", "score": _parse_score(line)}
                current_source = "hindsight"
            elif line.startswith("  🔗 [graph]"):
                if current:
                    results.append(current)
                current = {"source": "graph", "score": _parse_score(line)}
                current_source = "graph"
            elif line.startswith("  🧰 [skill]"):
                if current:
                    results.append(current)
                current = {"source": "skill", "score": _parse_score(line)}
                current_source = "skill"
            elif line.startswith("     🧰 技能:"):
                current["name"] = line.replace("     🧰 技能:", "").strip()
            elif line.startswith("     💡 加载:"):
                current["skill_view_cmd"] = line.replace("     💡 加载:", "").strip()
            elif line.startswith("     📝 说明:"):
                current["description"] = line.replace("     📝 说明:", "").strip()
            elif line.startswith("     🏷️  分类:"):
                current["category"] = line.replace("     🏷️  分类:", "").strip()
            elif line.startswith("     关联记忆:"):
                current["content"] = line.replace("     关联记忆:", "").strip()
            elif line.startswith("     路径:"):
                current["path"] = line.replace("     路径:", "").strip()
            elif line.startswith("     标题:"):
                current["title"] = line.replace("     标题:", "").strip()
            elif line.startswith("     关联实体:"):
                current["entity"] = line.replace("     关联实体:", "").strip()
            elif line.strip().startswith("·"):
                if "snippet" not in current:
                    current["snippet"] = []
                current["snippet"].append(line.strip())

        if current:
            results.append(current)

        return json.dumps({
            "success": True,
            "query": query,
            "source": source,
            "count": len(results),
            "results": results,
        }, ensure_ascii=False)

    except subprocess.TimeoutExpired:
        return json.dumps({
            "success": False,
            "error": "知识路由查询超时（30s）",
            "query": query,
            "source": source,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"知识路由异常: {e}",
            "query": query,
            "source": source,
        }, ensure_ascii=False)


def _parse_score(line: str) -> float:
    """从 '📖 [wiki] rel=0.62 ██' 中解析分数"""
    import re
    m = re.search(r"rel=([\d.]+)", line)
    return float(m.group(1)) if m else 0.0


# ── 注册为 Hermes 工具 ──
from tools.registry import registry  # noqa: E402

registry.register(
    name="knowledge_router",
    toolset="memory",
    schema={
        "name": "knowledge_router",
        "description": "统一知识检索：从 wiki 知识库 + hindsight 经验记忆 + graph 图扩散联想 + skill 技能推荐四层检索。"
                       "搜股票名/指标/概念时使用，自动联想关联实体。"
                       "相当于 session_search 的永久知识库版本。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，如 '卫星化学'、'ADX指标'、'BOLL和ADX的关系'",
                },
                "source": {
                    "type": "string",
                    "enum": ["auto", "all", "wiki", "hindsight", "graph", "skill"],
                    "description": "搜索范围: auto(自动路由) / all(全层+图扩散+skill推荐) / wiki / hindsight / graph(联想记忆) / skill(技能推荐)",
                    "default": "auto",
                },
            },
            "required": ["query"],
        },
    },
    handler=lambda args, **kw: knowledge_router(
        query=args.get("query", ""),
        source=args.get("source", "auto"),
    ),
    check_fn=check_requirements,
    requires_env=[],
)
logger.info("knowledge_router 工具已注册")
