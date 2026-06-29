"""
Instinct system bootstrap — monkey-patches AIAgent at import time.

Keeps instinct-specific code out of run_agent.py so that file can stay
100% in sync with upstream without losing the learned behavioral-pattern
injection (instincts system).

Pattern: ``run_agent.AIAgent._build_system_prompt`` is patched to append
the instinct-generated behavioral guidance after the normal system prompt.
This is a **post‑build hook**: the original method runs first, then the
instinct prompt (if available) is appended to the result string.

All gateway/CLI entry points should import this module exactly once during
startup — the patch is effective for every subsequent AIAgent instance.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def install() -> None:
    """Apply the instinct monkey-patch to ``AIAgent._build_system_prompt``.

    Safe to call multiple times — only the first call applies the patch.
    """
    global _PATCHED
    if _PATCHED:
        return

    from run_agent import AIAgent

    _orig = AIAgent._build_system_prompt

    def _patched_build_system_prompt(self, system_message: str = None) -> str:
        result = _orig(self, system_message=system_message)
        try:
            from agent.instincts import inject_instincts_prompt

            # Channel 1: tool reliability/performance — 200 chars
            tool_block = inject_instincts_prompt(threshold=0.9, max_chars=200,
                                                  domain_filter=None)
            # Channel 2: behavior rules — 200 chars for seeded CORE habits
            # NOTE: threshold=0.5 临时降低以便 knowledge_router_first(0.57) 注入 prompt，
            #       观察效果后再决定是否回调至 0.6
            behavior_block = inject_instincts_prompt(threshold=0.5, max_chars=200,
                                                      domain_filter="behavior")

            blocks = []
            if tool_block:
                blocks.append(tool_block)
            if behavior_block:
                blocks.append(behavior_block)
            if blocks:
                instinct_text = "\n\n".join(blocks)
                result = f"{result}\n\n{instinct_text}"
        except Exception as exc:
            logger.debug("Instinct prompt injection skipped: %s", exc)
        return result

    AIAgent._build_system_prompt = _patched_build_system_prompt

    # ── Patch 2: session-end decay for knowledge_router_first habit ──
    _orig_commit = AIAgent.commit_memory_session

    def _patched_commit_memory_session(self, messages: list = None) -> None:
        """Run session-end decay on old session_id, then original commit."""
        try:
            from agent.instincts import apply_session_end_decay
            old_sid = getattr(self, "session_id", None)
            if old_sid:
                apply_session_end_decay(old_sid)
        except Exception as exc:
            logger.debug("Session-end decay skipped: %s", exc)
        return _orig_commit(self, messages=messages)

    AIAgent.commit_memory_session = _patched_commit_memory_session
    _PATCHED = True
    logger.info("Instinct bootstrap installed — _build_system_prompt + commit_memory_session patched")
