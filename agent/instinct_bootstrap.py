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

            instinct = inject_instincts_prompt(threshold=0.9, max_chars=300)
            if instinct:
                result = f"{result}\n\n{instinct}"
        except Exception as exc:
            logger.debug("Instinct prompt injection skipped: %s", exc)
        return result

    AIAgent._build_system_prompt = _patched_build_system_prompt
    _PATCHED = True
    logger.info("Instinct bootstrap installed — AIAgent._build_system_prompt patched")
