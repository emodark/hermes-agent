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
import re
import shlex
import subprocess
import sys
import time

logger = logging.getLogger(__name__)

KNOWLEDGE_ROUTER = os.path.expanduser("~/.hermes/scripts/knowledge_router.py")
_TRIGGERED_SKILLS = set()  # 避免同会话重复触发同一 skill

# ── 降级监控 ──
_DEGRADATION_COUNT = 0
_DEGRADATION_LAST_RESET = 0.0
_DEGRADATION_RESET_AFTER = 300  # 300秒后重置计数


def _check_degradation():
    """检查降级频率，连续多次触发时输出 ERROR 告警。"""
    global _DEGRADATION_COUNT, _DEGRADATION_LAST_RESET
    now = time.time()
    if now - _DEGRADATION_LAST_RESET > _DEGRADATION_RESET_AFTER:
        _DEGRADATION_COUNT = 0
        _DEGRADATION_LAST_RESET = now
    _DEGRADATION_COUNT += 1
    if _DEGRADATION_COUNT >= 3:
        logger.error(
            "[DEGRADATION-HIGH] knowledge_router symbolic degraded %d times "
            "in %.0fs - check hindsight/network health",
            _DEGRADATION_COUNT, now - _DEGRADATION_LAST_RESET,
        )


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


def _detect_stock_code(cmd: str) -> str | None:
    """从 terminal 命令中检测6位疑似 A 股股票代码。

    Returns:
        匹配到的股票代码字符串，或 None。
    """
    if not cmd:
        return None
    # 匹配主流 A 股代码格式
    match = re.search(
        r'\b(?:60[0-9]{4}|00[0-9]{4}|30[0-9]{4}|68[0-9]{4}'
        r'|000[0-9]{3}|001[0-9]{3}|002[0-9]{3}|003[0-9]{3}'
        r'|200[0-9]{3}|900[0-9]{3}|4[0-9]{5}|8[0-9]{5})\b',
        cmd,
    )
    if match:
        return match.group(0)
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
    """terminal 的 pre-hook：执行前自动查知识路由 + 符号系统结构化知识。"""
    parts = []

    cmd = args.get("command", "")

    # Part 1: 技能推荐（原有逻辑）
    query = _extract_terminal_query(name, args)
    if query:
        output = _call_knowledge_router(query)
        if output:
            parts.append(f"【知识路由·自动检索】\n{output}")

    # Part 2: 符号系统结构化知识（当检测到个股代码时自动注入）
    stock_code = _detect_stock_code(cmd)
    if stock_code:
        symbolic_output = _call_knowledge_router_symbolic(stock_code)
        if symbolic_output:
            parts.append(f"【符号系统·结构化知识】\n{symbolic_output}")

    return "\n\n".join(parts) if parts else None


def _detect_stock_intent(goal: str) -> str | None:
    """检测goal是否包含股票分析意图，提取股票名/代码。

    Returns:
        提取到的股票代码或名称字符串，或 None（无匹配）。
    """
    if not goal:
        return None

    # 股票分析意图关键词
    stock_intent_keywords = [
        "股票分析",
        "持仓分析",
        "个股评估",
        "止损评估",
        "分析股票",
        "投资分析",
        "技术分析",
        "基本面分析",
    ]

    has_intent = any(kw in goal for kw in stock_intent_keywords)

    # 宽松匹配：包含常见股票动作/主体词
    loose_keywords = ["股票", "持仓", "评估", "止损", "分析", "加仓", "减仓", "买入", "卖出", "持有", "清仓"]
    if not has_intent:
        has_intent = any(kw in goal for kw in loose_keywords)

    if not has_intent:
        return None

    # 尝试提取股票代码（6位数字，以0/3/6开头）
    # 注意：不能依赖 \b 边界，因为中文在 Unicode 模式下也是 \w 字符
    code_match = re.search(
        r'(?<![0-9])([036]\d{5})(?![0-9])',
        goal,
    )
    if code_match:
        return code_match.group(1)

    # 尝试提取股票名称（汉字2-4字）
    # 优先提取跟在动作词后面的名称
    action_words = "分析|评估|查询|查找|看看|关注|止损|加仓|减仓|买入|卖出|持有|清仓"
    combined = re.search(
        rf'(?:{action_words})\s*([\u4e00-\u9fff]{{2,4}})',
        goal,
    )
    if combined:
        name = combined.group(1)
        if name not in ("股票", "分析", "持仓", "止损", "评估"):
            return name

    # 通用名称提取（2-4字的中文词，可能是股票名）
    # 跳过已知的非股票词
    skip_words = "股票|分析|持仓|止损|评估|投资|加仓|减仓|买入|卖出|持有|清仓|配置|修改|检查|查看|Redis|连接|config"
    name_match = re.search(
        rf'(?!(?:{skip_words}))([\u4e00-\u9fff]{{2,4}})',
        goal,
    )
    if name_match:
        name = name_match.group(1)
        if name not in ("股票", "分析", "持仓", "止损", "评估", "投资"):
            return name

    return None


def _call_knowledge_router_symbolic(query: str) -> str | None:
    """调 knowledge_router --symbolic-only，提取结构化摘要。

    先用 --all 全链检索（含 hindsight，可能慢），超时后降级为 wiki 只读。
    返回紧凑摘要（~200 字符），不返回完整 JSON。
    """
    import json as _json

    # 尝试0: 快速预检（纯本地skill匹配，5s超时）
    quick_check = _try_symbolic(query, ["--source=skill", "--symbolic-only"], timeout=5)

    # 不短路！继续尝试1，尝试0的结果在尝试1失败时作为保底

    # 尝试1: 全链检索（含 hindsight，冲突检测需要多源）
    pkg = _try_symbolic(query, ["--all", "--symbolic-only"], timeout=15)
    if pkg is None:
        # 尝试1超时/失败 → 降级为 wiki-only（快，但无冲突检测）
        _check_degradation()
        pkg = _try_symbolic(query, ["--source=wiki", "--symbolic-only"], timeout=5)

    # wiki也失败时，尝试用quick_check作为最后保底
    if pkg is None and quick_check is not None:
        pkg = quick_check

    # 完全无数据 → 返回最小知识包（而不是 None），给主控"无数据"提示
    if pkg is None:
        return (
            f"📊 股票: {query}\n"
            f"  ⚠️ 数据可用性: 知识库无此股票信息\n"
            f"  💡 建议: 派小金深挖（数据不足，需外部数据源补充）"
        )

    return _format_symbolic_summary(pkg, query)


def _format_symbolic_summary(pkg: dict, query: str) -> str:
    """格式化符号系统输出为带决策建议的紧凑摘要。

    覆盖3个场景：
    1. 派活前：数据可用性 + 股票类型 + 警告信号 → 决定自己干还是派活
    2. 审核时：冲突 + 可信度原因 → 判断报告可信度
    3. 自己动手时：预警信息 → 知道有什么坑
    """
    parts: list[str] = []

    # ── 1. 基础信息 ──
    stock_name = ""
    for f in pkg.get("facts", []):
        fn = f.get("fields", {})
        if isinstance(fn, dict):
            nv = fn.get("stock_name") or fn.get("code")
            if isinstance(nv, dict):
                stock_name = str(nv.get("value", ""))
            elif isinstance(nv, str):
                stock_name = nv
            if stock_name:
                break
    if stock_name and stock_name != query:
        parts.append(f"📊 {query} {stock_name}")
    else:
        parts.append(f"📊 股票: {query}")

    # ── 2. 股票类型（来自 stock_type_detector） — 决定派什么活 ──
    st = pkg.get("stock_type", {})
    if st and isinstance(st, dict):
        st_label = st.get("stock_type_label", "")
        st_advice = st.get("framework_advice", "")
        if st_label:
            parts.append(f"  类型: {st_label}")
        if st_advice:
            parts.append(f"  建议: {st_advice}")

    # ── 3. 数据可用性 — 决定自己快速过还是派活 ──
    facts = pkg.get("facts", [])
    conflicts = pkg.get("conflicts", [])
    rules = pkg.get("rules_triggered", [])

    # 检查是否有基本面数据
    has_fundamental = False
    for f in facts:
        fn = f.get("fields", {})
        if isinstance(fn, dict):
            for key in fn:
                if key in ("pe_ttm", "roe", "revenue", "net_profit"):
                    has_fundamental = True
                    break

    fact_count = len(facts)
    if fact_count == 0:
        parts.append("  ⚠️ 数据可用性: 无结构化事实 → 建议先查数据库")
    elif not has_fundamental:
        parts.append("  ⚠️ 数据可用性: 有限（仅%d条事实，无基本面指标）" % fact_count)
    else:
        parts.append("  ✅ 数据可用性: 完整（含基本面，%d条事实）" % fact_count)

    # ── 4. 已知风险/警告（来自 rule_engine）— 自己动手时的预警 ──
    seen_messages: set[str] = set()
    for r in rules:
        msg = r.get("message", "")
        if not msg or msg in seen_messages:
            continue
        seen_messages.add(msg)
        action = r.get("action", "")
        if action in ("warn", "reject", "flag_high"):
            parts.append(f"  🔴 {msg}")
        elif action in ("info", "flag"):
            parts.append(f"  🟡 {msg}")

    # ── 5. 信源冲突 — 审核时的信任判断 ──
    for c in conflicts[:2]:
        emoji = "🔴" if c.get("severity") == "error" else "🟡"
        field = c.get("field", "?")
        resolution = c.get("resolution", "")
        parts.append(f"  {emoji} 冲突: {field} → {resolution}")

    # ── 6. 叙事分歧 — 额外信号 ──
    narr = pkg.get("narrative", {})
    if narr and isinstance(narr, dict):
        primary = narr.get("primary_lens", "")
        if primary and primary != "none":
            tension = " ⚡分歧" if narr.get("tension_exists") else ""
            parts.append(f"  📰 叙事: {primary}{tension}")

    # ── 7. 可信度 + 原因 — 帮助判断是否该相信 ──
    confidence = pkg.get("confidence", "N/A")
    details = pkg.get("confidence_details", {}) or {}
    reasons: list[str] = []
    if details:
        fc = details.get("facts_count", 0)
        if fc > 0:
            reasons.append("%d条事实" % fc)
        else:
            reasons.append("无事实(惩罚)")
        cp = details.get("conflict_penalties", [])
        if cp:
            total_cp = sum(float(p.get("weight", 0)) for p in cp)
            reasons.append("冲突惩罚%.2f" % total_cp)
        rp = details.get("rule_penalties", [])
        if rp:
            total_rp = sum(float(p.get("weight", 0)) for p in rp)
            reasons.append("规则惩罚%.2f" % total_rp)
    reason_str = "，".join(reasons) if reasons else "无负面证据"
    parts.append("  📊 可信度: %s（%s）" % (confidence, reason_str))

    # ── 8. 派活建议（自动决策逻辑，供主控参考）──
    if has_fundamental and not any(r.get("action") in ("warn", "reject") for r in rules):
        parts.append("  💡 建议: 自己快速过（基本面齐全，无警告）")
    elif fact_count == 0:
        parts.append("  💡 建议: 派小金深挖（知识库无数据）")
    else:
        parts.append("  💡 建议: 派小金分析（数据有限或存在警告信号）")

    return "\n".join(parts)


def _try_symbolic(query: str, args: list[str], timeout: int = 25) -> dict | None:
    """调 knowledge_router --symbolic-only，返回知识包 dict。"""
    if not os.path.exists(KNOWLEDGE_ROUTER) or len(query.strip()) < 2:
        return None
    import json as _json

    try:
        result = subprocess.run(
            [sys.executable, KNOWLEDGE_ROUTER, query] + args,
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None

        output = result.stdout
        if "🧠 符号系统" not in output:
            return None

        json_part = output.split("🧠 符号系统")[-1]
        json_part = json_part.split("=" * 60)[-1].strip()
        return _json.loads(json_part)

    except subprocess.TimeoutExpired:
        logger.warning(
            "[DEGRADATION] knowledge_router --symbolic first attempt timeout, "
            "falling back to wiki-only: query=%s args=%s",
            query[:50], args,
        )
        return None
    except Exception as e:
        logger.debug("knowledge_router --symbolic error: %s", e)
        return None


# ── 会话级去重（已注释：1M 上下文窗口下无必要，每次注入仅~75 tokens）
# _TRIGGERED_STOCKS: set[str] = set()  # 已注入过符号知识的股票

def delegate_task_prehook(name: str, args: dict) -> str | None:
    """增强版 pre-hook：派单前自动查知识路由 + 符号系统。"""
    parts = []

    # 1. 原有技能推荐
    query = _extract_delegate_query(name, args)
    if query:
        output = _call_knowledge_router(query)
        if output:
            parts.append(f"【知识路由·技能推荐】\n{output}")

    # 2. 股票分析意图检测 → 符号系统结构化知识包（不再做会话级去重，
    #    1M 上下文窗口下每次 ~75 tokens 的注入成本可忽略）
    goal = args.get("goal", "")
    stock_ref = _detect_stock_intent(goal)
    if stock_ref:
        symbolic_output = _call_knowledge_router_symbolic(stock_ref)
        if symbolic_output:
            parts.append(f"【符号系统·结构化知识】\n{symbolic_output}")

    return "\n\n".join(parts) if parts else None


# ── 注册 pre-hooks ──
# 必须在模块顶层直接调用 registry.register_pre_hook() 才能被 AST 发现器扫描到
from tools.registry import registry  # noqa: E402

registry.register_pre_hook("terminal", terminal_prehook)
registry.register_pre_hook("delegate_task", delegate_task_prehook)
logger.info("knowledge_router pre-hooks registered (terminal, delegate_task)")
