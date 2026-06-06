"""Periodic process memory usage logging for the gateway.

Ported from cline/cline#10343 (src/standalone/memory-monitor.ts).

The gateway is a long-lived process that accumulates memory as it caches
agent instances, session transcripts, tool schemas, memory providers, MCP
connections, etc.  A slow leak in any of those subsystems is invisible
in a single log line — you only see it by watching RSS climb over hours.

This module emits a single structured ``[MEMORY] ...`` line every N
minutes (default 5) so maintainers investigating a suspected leak can
grep ``agent.log`` / ``gateway.log`` for a time series of RSS + Python
GC stats.  The timer runs in a background thread and shuts down cleanly
with the gateway.

Design notes (parity with the Cline port):
  * Grep-friendly single-line format beginning ``[MEMORY]``.
  * Reports **both** current RSS (from /proc/self/status VmRss) and
    peak RSS (from resource.getrusage().ru_maxrss) — the original Cline
    port used ru_maxrss which reports the process lifetime high-water
    mark, not current usage, making it useless for leak detection.
  * Final snapshot logged on shutdown so "last RSS before exit" is
    always in the log.
  * Baseline snapshot logged immediately on start.
  * Daemon thread — never blocks process exit.
  * Reads /proc/self/status directly (Linux-only, no extra deps) and
    falls back to ``psutil`` on other platforms.

Config: ``logging.memory_monitor`` in ``config.yaml`` — see
``hermes_cli/config.py`` for the defaults block.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_BYTES_TO_MB = 1024 * 1024

_monitor_thread: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None
_start_time: Optional[float] = None
_interval_seconds: float = 300.0  # 5 minutes
_lock = threading.Lock()


def _get_current_rss_mb() -> Optional[int]:
    """Return current process RSS in MB by reading /proc/self/status.

    This is the **real** current RSS, not the process lifetime peak.
    Falls back to psutil if /proc is not available (macOS, Windows).

    Why /proc/self/status instead of getrusage().ru_maxrss:
      ``ru_maxrss`` reports the **peak** RSS over the process lifetime
      (the high-water mark).  For a leak-detection monitor we need the
      **current** RSS so we can see the trend over time.  A process that
      peaked at 5GB during startup and settled at 2GB looks "leaking at
      5GB" if you use ru_maxrss, which is misleading.
    """
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # Format: "VmRSS:  1234567 kB"
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        return int(parts[1]) // 1024
    except (FileNotFoundError, IOError, ValueError, OSError):
        pass

    # Fallback: psutil (macOS, Windows, containers without /proc)
    try:
        import psutil  # type: ignore
        rss = psutil.Process(os.getpid()).memory_info().rss
        return int(rss / _BYTES_TO_MB)
    except Exception:
        return None


def _get_peak_rss_mb() -> Optional[int]:
    """Return process lifetime peak RSS in MB via resource.getrusage.

    ``ru_maxrss`` is the high-water mark.  We report it alongside
    current RSS so you can tell "still growing" from "startup spike".
    On Linux ru_maxrss is in KB, on macOS in bytes.
    """
    try:
        import resource
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(maxrss / _BYTES_TO_MB)
        return int(maxrss / 1024)
    except Exception:
        return None


def log_memory_usage(prefix: str = "") -> None:
    """Log current + peak memory usage in a grep-friendly ``[MEMORY] ...`` line.

    Safe to call on-demand from any thread at important lifecycle
    moments (after shutdown, after context compression, etc.).

    Parameters
    ----------
    prefix
        Optional extra tag inserted after ``[MEMORY]`` — e.g.
        ``"baseline"``, ``"shutdown"``.
    """
    current = _get_current_rss_mb()
    peak = _get_peak_rss_mb()
    uptime = int(time.monotonic() - _start_time) if _start_time else 0
    try:
        gc_counts = gc.get_count()
    except Exception:
        gc_counts = (0, 0, 0)
    try:
        thread_count = threading.active_count()
    except Exception:
        thread_count = 0

    tag = f"{prefix} " if prefix else ""

    if current is None:
        logger.info(
            "[MEMORY] %srss_cur=unavailable peak=%dMB gc=%s threads=%d uptime=%ds",
            tag, peak or 0, gc_counts, thread_count, uptime,
        )
    elif peak is not None:
        logger.info(
            "[MEMORY] %srss_cur=%dMB peak=%dMB gc=%s threads=%d uptime=%ds",
            tag, current, peak, gc_counts, thread_count, uptime,
        )
    else:
        logger.info(
            "[MEMORY] %srss_cur=%dMB peak=unavailable gc=%s threads=%d uptime=%ds",
            tag, current, gc_counts, thread_count, uptime,
        )


def _monitor_loop(stop_event: threading.Event, interval: float) -> None:
    """Background thread body — log every ``interval`` seconds until stopped."""
    while not stop_event.wait(interval):
        try:
            log_memory_usage()
        except Exception as e:
            logger.debug("Memory monitor iteration failed: %s", e)


def start_memory_monitoring(interval_seconds: float = 300.0) -> bool:
    """Start periodic memory usage logging in a daemon thread.

    Logs immediately to capture a baseline, then every ``interval_seconds``.
    Safe to call multiple times — subsequent calls are no-ops while the
    first monitor is still running.

    Parameters
    ----------
    interval_seconds
        How often to log.  Default 300s (5 minutes), matching the
        upstream cline/cline implementation.

    Returns
    -------
    bool
        True if a fresh monitor thread was started, False if one was
        already running or if memory introspection isn't available.
    """
    global _monitor_thread, _stop_event, _start_time, _interval_seconds

    with _lock:
        if _monitor_thread is not None and _monitor_thread.is_alive():
            return False

        if _get_current_rss_mb() is None:
            logger.warning(
                "[MEMORY] Memory monitoring unavailable: neither /proc/self/status "
                "nor psutil could read current RSS — skipping periodic logging.",
            )
            return False

        _start_time = time.monotonic()
        _interval_seconds = float(interval_seconds)
        _stop_event = threading.Event()

        log_memory_usage(prefix="baseline")

        _monitor_thread = threading.Thread(
            target=_monitor_loop,
            args=(_stop_event, _interval_seconds),
            name="gateway-memory-monitor",
            daemon=True,
        )
        _monitor_thread.start()

        logger.info(
            "[MEMORY] Periodic memory monitoring started (interval: %ds)",
            int(_interval_seconds),
        )
        return True


def stop_memory_monitoring(timeout: float = 2.0) -> None:
    """Stop the monitor thread and log a final snapshot.

    Safe to call even if ``start_memory_monitoring()`` was never called.
    """
    global _monitor_thread, _stop_event

    with _lock:
        if _stop_event is None or _monitor_thread is None:
            return

        try:
            log_memory_usage(prefix="shutdown")
        except Exception:
            pass

        _stop_event.set()
        thread = _monitor_thread
        _monitor_thread = None
        _stop_event = None

    try:
        thread.join(timeout=timeout)
    except Exception:
        pass

    logger.info("[MEMORY] Periodic memory monitoring stopped")


def is_running() -> bool:
    """True if the background monitor thread is alive."""
    with _lock:
        return _monitor_thread is not None and _monitor_thread.is_alive()
