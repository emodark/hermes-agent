#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────
# Rebase Parameter Audit — 在 rebase/update 后自动检查
# AIAgent.__init__ 和 FailoverReason 枚举是否缺上游新增参数
#
# 用法：cd ~/.hermes/hermes-agent && bash scripts/rebase-parameter-audit.sh
#
# 在 main-local rebase 到 main 之后、提交之前运行。
# 输出为空 = 无遗漏；有输出 = 列出缺失的参数和枚举成员。
# ───────────────────────────────────────────────────────────
set -euo pipefail

MISSING=0

# ── Helper: 从文件中提取 AIAgent.__init__ 参数列表 ──
extract_aia_init_params() {
    local file="$1"
    # 仅提取 class AIAgent 范围内的参数
    sed -n '/^class AIAgent:/,/^class /p' "$file" \
        | grep -A300 'def __init__(' \
        | sed -n '1,/^    def /p' \
        | grep -E '^\s+\w+:' \
        | sed 's/^\s*//;s/:.*//' \
        | sort
}

extract_enum_members() {
    local file="$1"
    grep -E '^\s+\w+ = "' "$file" \
        | sed 's/^\s*//' \
        | sed 's/ = .*//' \
        | sort
}

# ── 1) AIAgent.__init__ 参数审计 ────────────────────────
echo "═══════════════════════════════════════════════════"
echo "  AIAgent.__init__() 参数完整性审计"
echo "═══════════════════════════════════════════════════"

# 当前 HEAD 的 run_agent.py
LOCAL_FILE="run_agent.py"

# 从 upstream/main 取最新版
UPSTREAM_INIT=$(mktemp)
LOCAL_INIT=$(mktemp)

extract_aia_init_params <(git show upstream/main:run_agent.py 2>/dev/null) > "$UPSTREAM_INIT" || {
    echo "⚠ 无法读取 upstream/main:run_agent.py — 跳过上游比对"
    rm -f "$UPSTREAM_INIT" "$LOCAL_INIT"
    exit 0
}
extract_aia_init_params "$LOCAL_FILE" > "$LOCAL_INIT"

# 上游有但我们没有的 = 缺失参数
UPSTREAM_ONLY=$(comm -23 "$UPSTREAM_INIT" "$LOCAL_INIT" 2>/dev/null | grep -v '^$' || true)
if [ -n "$UPSTREAM_ONLY" ]; then
    echo "⚠ 以下参数在 upstream/main 中已存在但本地缺失："
    echo "$UPSTREAM_ONLY" | while read -r param; do
        # 显示类型和默认值
        TYPE=$(grep -E "^\s+${param}:" <(git show upstream/main:run_agent.py) | head -1 | sed 's/.*: *//' | sed 's/ *=.*//')
        DEFAULT=$(grep -E "^\s+${param}:" <(git show upstream/main:run_agent.py) | head -1 | grep -oP '= \K.*' || echo "None")
        echo "   - $param: $TYPE = $DEFAULT"
    done
    MISSING=1
else
    echo "✅ AIAgent.__init__ 参数完全对齐"
fi

rm -f "$UPSTREAM_INIT" "$LOCAL_INIT"

# ── 2) FailoverReason 枚举审计 ─────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  FailoverReason 枚举完整性审计"
echo "═══════════════════════════════════════════════════"

UPSTREAM_ENUM=$(mktemp)
LOCAL_ENUM=$(mktemp)

extract_enum_members <(git show upstream/main:agent/error_classifier.py 2>/dev/null) > "$UPSTREAM_ENUM" || {
    echo "⚠ 无法读取 upstream/main:agent/error_classifier.py"
    rm -f "$UPSTREAM_ENUM" "$LOCAL_ENUM"
    exit 0
}
extract_enum_members "agent/error_classifier.py" > "$LOCAL_ENUM"

ENUM_MISSING=$(comm -23 "$UPSTREAM_ENUM" "$LOCAL_ENUM" 2>/dev/null | grep -v '^$' || true)
if [ -n "$ENUM_MISSING" ]; then
    echo "⚠ 以下枚举成员在 upstream/main 中已存在但本地缺失："
    echo "$ENUM_MISSING" | while read -r member; do
        echo "   - $member"
    done
    MISSING=1
else
    echo "✅ FailoverReason 枚举完全对齐"
fi

rm -f "$UPSTREAM_ENUM" "$LOCAL_ENUM"

# ── 3) hermes_state.py 薄壳审计 ────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  hermes_state.py 薄壳符号审计"
echo "═══════════════════════════════════════════════════"

LOCAL_THIN=$(grep -c "_lmdb_all\|from hermes_state_lmdb" "hermes_state.py" 2>/dev/null || echo 0)
if [ "$LOCAL_THIN" -gt 0 ]; then
    echo "✅ hermes_state.py 薄壳结构完好"
else
    echo "⚠ hermes_state.py 未检测到薄壳导入模式 — 可能被覆盖"
fi

# ── 4) 总结 ────────────────────────────────────────────
echo ""
if [ "$MISSING" -eq 1 ]; then
    echo "❌ 发现缺失参数/枚举 — 请在提交前修复"
    echo "   修复方式：将缺失参数添加到对应文件的 __init__ / FailoverReason 中"
    exit 1
else
    echo "✅ 全部对齐 — 可以安全提交"
    exit 0
fi
