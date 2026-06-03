"""
Instinct System — automatic behavior pattern learning for Hermes Agent.

Architecture:
  Observations (JSONL) → Analyzer (cluster+score) → Instincts (YAML) → System Prompt injection

Inspired by everything-claude-code's Instincts v2, but adapted for Hermes:
  - Uses hindsight-compatible tagging
  - Confidence scoring with evidence tracking
  - Domain-scoped + auto-promotion to global
  - Clean YAML output for human review
"""

import json
import logging
import os
import time
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
_INSTINCTS_DIR = _HERMES_HOME / "instincts"
_OBSERVATIONS_FILE = _INSTINCTS_DIR / "observations.jsonl"
_INSTINCTS_FILE = _INSTINCTS_DIR / "instincts.yaml"

# Minimum observations to attempt analysis
_MIN_OBSERVATIONS_FOR_ANALYSIS = 20

# Rolling window — observations beyond this count are trimmed (oldest dropped)
# Keeps the JSONL file bounded. 10K observations ≈ 3MB on disk.
_MAX_OBSERVATIONS = 10000
_TRIM_CHECK_INTERVAL = 50  # check file size every N writes (probabilistic, not per-write IO)
_write_counter = 0  # module-level counter for probabilistic trim check


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def _ensure_dir():
    _INSTINCTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_truncate(obj: Any, max_chars: int = 200) -> str:
    """Safely truncate a value for observation logging."""
    text = json.dumps(obj, ensure_ascii=False, default=str)
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def _maybe_trim_observations(max_lines: int = _MAX_OBSERVATIONS) -> None:
    """Trim observations file to keep only the most recent max_lines.

    Called probabilistically from record_observation() to keep the JSONL
    file bounded. Uses 20% hysteresis to avoid thrashing.
    Non-blocking — errors are caught and logged at DEBUG level.
    """
    try:
        if not _OBSERVATIONS_FILE.exists():
            return

        with open(_OBSERVATIONS_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) <= int(max_lines * 1.2):
            return

        with open(_OBSERVATIONS_FILE, "w", encoding="utf-8") as f:
            f.writelines(lines[-max_lines:])

        logger.info(
            f"Trimmed observations to {max_lines} lines (was {len(lines)})"
        )
    except Exception as e:
        logger.debug(f"Instinct trim failed (non-critical): {e}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def record_observation(
    tool_name: str,
    args: Dict[str, Any],
    result: str,
    duration_ms: int,
    session_id: str = "",
    user_message_context: str = "",
) -> None:
    """Record a tool call observation for instinct learning.

    Called by model_tools.py handle_function_call() after post_tool_call hook.
    Non-blocking append to JSONL — no sync/fsync, crash-safe enough for stats.
    """
    try:
        _ensure_dir()

        # Determine if tool call succeeded
        success = True
        result_preview = ""
        if isinstance(result, str):
            result_preview = result[:200]
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict) and "error" in parsed:
                    success = False
                    result_preview = f"ERROR: {parsed['error'][:200]}"
            except (json.JSONDecodeError, TypeError):
                pass

        # Extract a simplified args signature (tool name + key arg types)
        arg_keys = sorted(args.keys()) if isinstance(args, dict) else []
        key_args = {}
        for k in arg_keys[:5]:  # top 5 args
            v = args.get(k)
            if isinstance(v, str) and len(v) > 80:
                key_args[k] = v[:80] + "..."
            elif isinstance(v, (str, int, float, bool)):
                key_args[k] = v
            else:
                key_args[k] = type(v).__name__

        observation = {
            "tool": tool_name,
            "args_preview": key_args,
            "arg_keys": arg_keys,
            "success": success,
            "duration_ms": duration_ms,
            "session_id": session_id or "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Append to JSONL
        with open(_OBSERVATIONS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(observation, ensure_ascii=False) + "\n")

        # Probabilistic trim check — keeps file bounded without per-write IO
        global _write_counter
        _write_counter += 1
        if _write_counter % _TRIM_CHECK_INTERVAL == 0:
            _maybe_trim_observations()

    except Exception as e:
        logger.debug(f"Instinct observation failed (non-critical): {e}")


def get_high_confidence_instincts(threshold: float = 0.7, max_instincts: int = 10) -> List[Dict]:
    """Load instincts and return those above confidence threshold."""
    instincts = _load_instincts()
    return [i for i in instincts if i.get("confidence", 0) >= threshold][:max_instincts]


# ── Prompt cache ────────────────────────────────────────
_INSTINCT_PROMPT_CACHE: Dict[str, str] = {}
_INSTINCT_PROMPT_CACHE_TS: float = 0
_INSTINCT_PROMPT_CACHE_TTL: float = 900  # 15 分钟


def inject_instincts_prompt(threshold: float = 0.7, max_chars: int = 400) -> str:
    """Build a system prompt block from high-confidence instincts.

    Args:
        threshold: Minimum confidence to include (0.0-1.0)
        max_chars: Hard character limit for the entire block.
                  Longer content is trimmed from the lowest-confidence end.

    Returns empty string if no high-confidence instincts exist.
    Uses 15-minute in-memory cache keyed by max_chars to avoid repeated YAML IO.
    """
    global _INSTINCT_PROMPT_CACHE, _INSTINCT_PROMPT_CACHE_TS
    cache_key = f"mc={max_chars}"
    now = time.monotonic()
    if cache_key in _INSTINCT_PROMPT_CACHE and (now - _INSTINCT_PROMPT_CACHE_TS) < _INSTINCT_PROMPT_CACHE_TTL:
        return _INSTINCT_PROMPT_CACHE[cache_key]

    instincts = get_high_confidence_instincts(threshold, max_instincts=20)
    if not instincts:
        _INSTINCT_PROMPT_CACHE[cache_key] = ""
        _INSTINCT_PROMPT_CACHE_TS = now
        return ""

    # Sort by confidence descending, then build incrementally
    instincts.sort(key=lambda i: i.get("confidence", 0), reverse=True)

    header = "<instincts>\nThe following behavior patterns have been observed:\n"
    footer = "\n</instincts>"
    # Reserve space: len(header) + len(footer) + overhead per line
    reserved = len(header) + len(footer) + 20
    budget = max_chars - reserved
    if budget <= 0:
        _INSTINCT_PROMPT_CACHE[cache_key] = ""
        _INSTINCT_PROMPT_CACHE_TS = now
        return ""

    blocks = []
    for instinct in instincts:
        name = instinct.get("name", "")
        desc = instinct.get("description", "")
        conf = instinct.get("confidence", 0)
        domain = instinct.get("domain", "general")
        line = f"- [{domain}] {desc} (confidence: {conf:.0%})"
        # Check if adding this line would exceed budget
        candidate = "\n".join(blocks + [line])
        if len(candidate) > budget:
            break  # stop adding — ran out of chars
        blocks.append(line)

    if not blocks:
        _INSTINCT_PROMPT_CACHE[cache_key] = ""
        _INSTINCT_PROMPT_CACHE_TS = now
        return ""

    prompt = "\n".join(["", header, *blocks, footer])
    _INSTINCT_PROMPT_CACHE[cache_key] = prompt
    _INSTINCT_PROMPT_CACHE_TS = now
    return prompt


# ---------------------------------------------------------------------------
# Observation analysis → instinct generation
# ---------------------------------------------------------------------------

def analyze_observations() -> List[Dict]:
    """Read all observations and cluster into candidate instincts.

    Returns list of instinct candidates with confidence scores.
    """
    observations = _load_observations()
    if len(observations) < _MIN_OBSERVATIONS_FOR_ANALYSIS:
        logger.info(
            f"Instinct analyzer: only {len(observations)} observations "
            f"(need {_MIN_OBSERVATIONS_FOR_ANALYSIS})"
        )
        return []

    # --- Pattern 1: Most-used tools ---
    tool_counter = Counter(o["tool"] for o in observations)
    total_obs = len(observations)
    tool_instincts = _analyze_tool_frequencies(tool_counter, total_obs)

    # --- Pattern 2: Tool usage patterns (which args used most) ---
    arg_pattern_instincts = _analyze_arg_patterns(observations)

    # --- Pattern 3: Error/success patterns ---
    error_instincts = _analyze_error_patterns(observations)

    # --- Pattern 4: Session timing patterns ---
    timing_instincts = _analyze_timing_patterns(observations)

    all_candidates = tool_instincts + arg_pattern_instincts + error_instincts + timing_instincts

    # Merge duplicates by name (keep highest confidence)
    merged = {}
    for c in all_candidates:
        name = c["name"]
        if name not in merged or c["confidence"] > merged[name]["confidence"]:
            merged[name] = c

    return sorted(merged.values(), key=lambda x: x["confidence"], reverse=True)


def _analyze_tool_frequencies(tool_counter: Counter, total: int) -> List[Dict]:
    """Analyze which tools are used most frequently."""
    instincts = []
    top_tools = tool_counter.most_common(5)
    for tool, count in top_tools:
        ratio = count / total
        if ratio > 0.10:  # tool used >10% of all calls
            instincts.append({
                "name": f"frequent-tool-{tool}",
                "description": f"Tool '{tool}' is frequently used ({count}/{total} calls, {ratio:.0%})",
                "domain": "tool-usage",
                "confidence": min(ratio, 0.95),
                "evidence": {"tool": tool, "count": count, "ratio": round(ratio, 3)},
                "scope": "session",
            })
    return instincts


def _analyze_arg_patterns(observations: List[Dict]) -> List[Dict]:
    """Analyze common argument patterns in tool calls."""
    instincts = []

    # Cluster by (tool, arg_keys) to find repeated patterns
    pattern_counts = defaultdict(lambda: {"count": 0, "samples": []})
    for obs in observations:
        tool = obs.get("tool", "")
        arg_keys = tuple(obs.get("arg_keys", []))
        key = (tool,) + arg_keys
        pattern_counts[key]["count"] += 1
        if len(pattern_counts[key]["samples"]) < 3:
            pattern_counts[key]["samples"].append({
                "args": obs.get("args_preview", {}),
                "success": obs.get("success", True),
            })

    total = len(observations)
    for (tool, *arg_keys), data in pattern_counts.items():
        count = data["count"]
        if count >= 3 and count / total > 0.03:
            arg_list = list(arg_keys) if arg_keys else []
            if arg_list:
                instincts.append({
                    "name": f"pattern-{tool}-{'-'.join(arg_list[:3])}",
                    "description": f"Tool '{tool}' commonly called with args: {', '.join(arg_list)}",
                    "domain": "tool-pattern",
                    "confidence": min(count / total * 3, 0.85),
                    "evidence": {
                        "tool": tool,
                        "arg_keys": arg_list,
                        "count": count,
                        "samples": data["samples"],
                    },
                    "scope": "session",
                })

    return instincts


def _analyze_error_patterns(observations: List[Dict]) -> List[Dict]:
    """Analyze tools with high error rates."""
    tool_stats = defaultdict(lambda: {"total": 0, "errors": 0})
    for obs in observations:
        tool = obs.get("tool", "")
        tool_stats[tool]["total"] += 1
        if not obs.get("success", True):
            tool_stats[tool]["errors"] += 1

    instincts = []
    for tool, stats in tool_stats.items():
        if stats["total"] >= 5:
            error_rate = stats["errors"] / stats["total"]
            if error_rate > 0.20:
                instincts.append({
                    "name": f"error-prone-{tool}",
                    "description": f"Tool '{tool}' has {error_rate:.0%} error rate ({stats['errors']}/{stats['total']})",
                    "domain": "reliability",
                    "confidence": min(error_rate, 0.9),
                    "evidence": {
                        "tool": tool,
                        "error_rate": round(error_rate, 3),
                        "error_count": stats["errors"],
                        "total": stats["total"],
                    },
                    "scope": "session",
                })

    return instincts


def _analyze_timing_patterns(observations: List[Dict]) -> List[Dict]:
    """Analyze tools with unusual timing (fast/slow)."""
    tool_times = defaultdict(list)
    for obs in observations:
        tool = obs.get("tool", "")
        duration = obs.get("duration_ms", 0)
        if duration > 0:
            tool_times[tool].append(duration)

    instincts = []
    for tool, times in tool_times.items():
        if len(times) >= 5:
            avg = sum(times) / len(times)
            if avg > 5000:  # >5s average
                instincts.append({
                    "name": f"slow-tool-{tool}",
                    "description": f"Tool '{tool}' is consistently slow (avg {avg:.0f}ms, {len(times)} calls)",
                    "domain": "performance",
                    "confidence": min(0.5 + avg / 30000, 0.85),
                    "evidence": {
                        "tool": tool,
                        "avg_duration_ms": round(avg),
                        "sample_count": len(times),
                    },
                    "scope": "session",
                })

    return instincts


def save_instincts(instincts: List[Dict]) -> None:
    """Save candidate instincts to YAML file.

    Merges with existing instincts — keeps old ones with higher confidence,
    adds new ones, removes stale ones (not observed recently).
    """
    _ensure_dir()
    existing = _load_instincts()
    existing_map = {i["name"]: i for i in existing}

    for candidate in instincts:
        name = candidate["name"]
        if name in existing_map:
            old = existing_map[name]
            # Keep higher confidence, but decay old ones
            if candidate["confidence"] > old.get("confidence", 0):
                candidate["scope"] = "global" if candidate["confidence"] > 0.8 else old.get("scope", "session")
                existing_map[name] = candidate
        else:
            existing_map[name] = candidate

    # Remove instincts with confidence < 0.1 (stale)
    final = [v for v in existing_map.values() if v.get("confidence", 0) >= 0.1]

    # Write as YAML manually (avoids pyyaml dependency)
    lines = ["# Hermes Agent Instincts — auto-generated, do not edit manually", f"# Generated: {datetime.now(timezone.utc).isoformat()}", f"# Total: {len(final)} instincts", "", "instincts:"]
    for instinct in sorted(final, key=lambda x: x.get("confidence", 0), reverse=True):
        lines.append(f"  - name: {instinct.get('name', 'unknown')}")
        lines.append(f"    description: \"{instinct.get('description', '')}\"")
        lines.append(f"    domain: {instinct.get('domain', 'general')}")
        lines.append(f"    confidence: {instinct.get('confidence', 0):.2f}")
        lines.append(f"    scope: {instinct.get('scope', 'session')}")
        ev = instinct.get("evidence", {})
        lines.append(f"    evidence: {json.dumps(ev, ensure_ascii=False)}")

    with open(_INSTINCTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Saved {len(final)} instincts to {_INSTINCTS_FILE}")


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------

def _load_observations() -> List[Dict]:
    """Load all observations from JSONL file."""
    if not _OBSERVATIONS_FILE.exists():
        return []
    observations = []
    try:
        with open(_OBSERVATIONS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        observations.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.warning(f"Failed to load observations: {e}")
    return observations


def _load_instincts() -> List[Dict]:
    """Load instincts from YAML file."""
    if not _INSTINCTS_FILE.exists():
        return []
    instincts = []
    try:
        with open(_INSTINCTS_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        # Simple YAML parser for our format
        current = None
        for line in content.split("\n"):
            line_stripped = line.strip()
            if line_stripped == "instincts:":
                continue
            if line_stripped.startswith("- name:"):
                if current:
                    instincts.append(current)
                current = {"name": line_stripped.split(":", 1)[1].strip()}
            elif current:
                if ":" in line_stripped:
                    key, value = line_stripped.split(":", 1)
                    key = key.strip()
                    value = value.strip().strip('"')
                    if key in ("description", "domain", "scope"):
                        current[key] = value
                    elif key == "confidence":
                        try:
                            current[key] = float(value)
                        except ValueError:
                            current[key] = 0.0
                    elif key == "evidence":
                        try:
                            current[key] = json.loads(value)
                        except json.JSONDecodeError:
                            current[key] = {"raw": value}
        if current:
            instincts.append(current)
    except Exception as e:
        logger.warning(f"Failed to load instincts: {e}")
    return instincts


def get_observation_count() -> int:
    """Return the number of observations recorded."""
    return len(_load_observations())
