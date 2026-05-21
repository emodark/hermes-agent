#!/usr/bin/env python3
"""
knowledge_prehooks.py — 自动知识路由 pre-hook

在特定工具（delegate_task, terminal）执行前自动调 knowledge_router，
把知识上下文+skill推荐注入到工具结果中，让主控 Agent 在做决策前
自动获取相关背景。

注册机制：
  tools/registry.py 的 register_pre_hook() + dispatch() pre-hook 执行
"""
import json
import logging
import os
import shlex
import subprocess
import sys

logger = logging.getLogger(__name__)

KNOWLEDGE_ROUTER = os.path.expanduser("~/.hermes/scripts/knowledge_router.py")
_TRIGGERED_SKILLS = set()  # 避免同会话重复触发同一 skill


def _call_knowledge_router(query: str) -> str | None:
    """调 knowledge_router --skill，返回技能推荐结果。

    使用 --skill 模式（纯本地匹配，无 hindsight API 调用），
    确保 pre-hook 轻量快速。
    """
    if not os.path.exists(KNOWLEDGE_ROUTER) or len(query.strip()) < 5:
        return None
    try:
        result = subprocess.run(
            [sys.executable, KNOWLEDGE_ROUTER, query, "--skill"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None

        output = result.stdout.strip()
        # 检查是否有实际结果（跳过只有 header 的情况）
        lines = output.split("\n")
        result_lines = [l for l in lines if l.startswith("  ") and ("rel=" in l or "🧰" in l)]
        if not result_lines:
            return None

        return output
    except subprocess.TimeoutExpired:
        logger.debug("knowledge_router timeout for query: %s", query[:50])
        return None
    except Exception as e:
        logger.debug("knowledge_router error: %s", e)
        return None


def _extract_terminal_query(name: str, args: dict) -> str | None:
    """从 terminal 命令中提取知识路由查询。

    只对"有意义的操作"触发——非 trivial 命令。
    返回查询字符串，或 None（跳过）。
    """
    cmd = args.get("command", "")
    if not cmd or len(cmd) < 20:
        return None

    # 跳过纯浏览/文件操作
    skip_patterns = [
        "cd ", "ls ", "cat ", "head ", "tail ", "echo ", "source ",
        "which ", "pwd ", "clear", "env ", "export ",
    ]
    for pat in skip_patterns:
        if cmd.strip().startswith(pat):
            return None

    # 从命令中提取有意义的查询词
    # Python 脚本执行 → 用脚本名做查询
    if "python3" in cmd or "python" in cmd:
        # 提取文件名
        for token in shlex.split(cmd):
            if token.endswith(".py") and "/" in token:
                return token.rsplit("/", 1)[-1].replace(".py", "").replace("_", " ")
            elif token.endswith(".py"):
                return token.replace(".py", "").replace("_", " ")
        return None

    # git 操作
    if cmd.startswith("git"):
        return cmd[:80]

    return None


def _extract_delegate_query(name: str, args: dict) -> str | None:
    """从 delegate_task 参数中提取查询。"""
    goal = args.get("goal", "")
    if not goal or len(goal.strip()) < 10:
        return None
    # 取 goal 前 100 字符作为查询
    return goal.strip()[:100]


def terminal_prehook(name: str, args: dict) -> str | None:
    """terminal 的 pre-hook：执行前自动查知识路由。"""
    query = _extract_terminal_query(name, args)
    if not query:
        return None
    output = _call_knowledge_router(query)
    if output:
        return f"【知识路由·自动检索】\n{output}"
    return None


def delegate_task_prehook(name: str, args: dict) -> str | None:
    """delegate_task 的 pre-hook：派单前自动查知识路由。"""
    query = _extract_delegate_query(name, args)
    if not query:
        return None
    output = _call_knowledge_router(query)
    if output:
        return f"【知识路由·自动检索  |  查询: {query}】\n{output}"
    return None


# ── 注册 pre-hooks ──
# 必须在模块顶层直接调用 registry.register_pre_hook() 才能被 AST 发现器扫描到
from tools.registry import registry  # noqa: E402

registry.register_pre_hook("terminal", terminal_prehook)
registry.register_pre_hook("delegate_task", delegate_task_prehook)
logger.info("knowledge_router pre-hooks registered (terminal, delegate_task)")
