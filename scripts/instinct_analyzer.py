#!/usr/bin/env python3
"""
Instinct Analyzer — reads tool observations, clusters patterns, updates instincts.

Usage:
    python3 scripts/analyze_instincts.py          # analyze and update
    python3 scripts/analyze_instincts.py --status  # just show stats
    python3 scripts/analyze_instincts.py --prompt  # show instinct prompt block
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure we can import from the Hermes agent package
_HERMES_AGENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERMES_AGENT))

from agent.instincts import (
    analyze_observations,
    get_high_confidence_instincts,
    get_observation_count,
    inject_instincts_prompt,
    save_instincts,
    _load_observations,
    _OBSERVATIONS_FILE,
    _INSTINCTS_FILE,
)


def cmd_status():
    """Show observation and instinct statistics."""
    obs_count = get_observation_count()
    instincts = get_high_confidence_instincts(threshold=0.0)

    print(f"{'='*60}")
    print(f"  Instinct System Status")
    print(f"{'='*60}")
    print(f"  Observations file:  {_OBSERVATIONS_FILE}")
    print(f"  Total observations: {obs_count}")
    print(f"  Instincts file:     {_INSTINCTS_FILE}")
    print(f"  Total instincts:    {len(instincts)}")
    print()

    if instincts:
        print(f"  {'Instinct':<40} {'Conf':<8} {'Domain':<16}")
        print(f"  {'-'*40} {'-'*8} {'-'*16}")
        for i in instincts:
            name = i.get("name", "?")[:38]
            conf = i.get("confidence", 0)
            domain = i.get("domain", "?")[:14]
            print(f"  {name:<40} {conf:<8.0%} {domain:<16}")

    print()
    if obs_count < 20:
        print(f"  ⚠️  Need {20 - obs_count} more observations to trigger analysis.")
        print(f"     (20 minimum; keep using the agent normally)")
    else:
        print(f"  ✅ Enough observations for analysis. Run with no flags to update instincts.")


def cmd_analyze():
    """Run analysis and update instincts."""
    print(f"  Analyzing {get_observation_count()} observations...")
    candidates = analyze_observations()

    if not candidates:
        print("  No instinct candidates found.")
        return

    print(f"  Found {len(candidates)} instinct candidates:")
    for c in candidates:
        print(f"    • {c['name']:<45} conf={c['confidence']:.0%}  [{c['domain']}]")

    save_instincts(candidates)
    print(f"\n  ✅ Instincts saved to {_INSTINCTS_FILE}")

    # Show what would be injected
    prompt = inject_instincts_prompt(threshold=0.7)
    if prompt:
        print(f"\n  System prompt block ({len(prompt)} chars):")
        for line in prompt.split("\n"):
            print(f"    {line}")


def cmd_prompt():
    """Print the instinct system prompt block."""
    prompt = inject_instincts_prompt(threshold=0.7)
    if prompt:
        print(prompt)
    else:
        print("<!-- no high-confidence instincts -->")


def cmd_observations():
    """Dump recent observations."""
    obs = _load_observations()
    print(f"Total: {len(obs)} observations\n")
    for i, o in enumerate(obs[-30:]):  # last 30
        tool = o.get("tool", "?")
        success = "✅" if o.get("success", True) else "❌"
        duration = o.get("duration_ms", 0)
        args = o.get("args_preview", {})
        print(f"  {i+1:3d}. {success} {tool:<20} {duration:>6}ms  args={json.dumps(args, ensure_ascii=False)[:60]}")


def main():
    parser = argparse.ArgumentParser(description="Instinct System Analyzer")
    parser.add_argument("--status", action="store_true", help="Show stats only")
    parser.add_argument("--prompt", action="store_true", help="Print instinct prompt block")
    parser.add_argument("--observations", action="store_true", help="Dump recent observations")
    args = parser.parse_args()

    if args.status:
        cmd_status()
    elif args.prompt:
        cmd_prompt()
    elif args.observations:
        cmd_observations()
    else:
        cmd_analyze()


if __name__ == "__main__":
    main()
