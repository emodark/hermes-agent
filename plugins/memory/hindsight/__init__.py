"""Hindsight memory plugin — MemoryProvider interface.

Long-term memory with knowledge graph, entity resolution, and multi-strategy
retrieval. Supports cloud (API key) and local modes.

Configurable request timeout via HINDSIGHT_TIMEOUT env var or config.json.
Configurable embedded daemon idle timeout via HINDSIGHT_IDLE_TIMEOUT env var
or config.json idle_timeout.

Original PR #1811 by benfrank241, adapted to MemoryProvider ABC.

Config via environment variables:
  HINDSIGHT_API_KEY                — API key for Hindsight Cloud
  HINDSIGHT_BANK_ID                — memory bank identifier (default: hermes)
  HINDSIGHT_BUDGET                 — recall budget: low/mid/high (default: mid)
  HINDSIGHT_API_URL                — API endpoint
  HINDSIGHT_MODE                   — cloud or local (default: cloud)
  HINDSIGHT_TIMEOUT                — API request timeout in seconds (default: 120)
  HINDSIGHT_IDLE_TIMEOUT           — embedded daemon idle timeout seconds; 0 disables shutdown (default: 300)
  HINDSIGHT_RETAIN_TAGS            — comma-separated tags attached to retained memories
  HINDSIGHT_RETAIN_SOURCE          — metadata source value attached to retained memories
  HINDSIGHT_RETAIN_USER_PREFIX     — label used before user turns in retained transcripts
  HINDSIGHT_RETAIN_ASSISTANT_PREFIX — label used before assistant turns in retained transcripts

Or via $HERMES_HOME/hindsight/config.json (profile-scoped), falling back to
~/.hindsight/config.json (legacy, shared) for backward compatibility.
"""

from __future__ import annotations

import asyncio
import atexit
import importlib
import json
import logging
import os
import queue
import re
import threading

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home
from tools.registry import tool_error
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "https://api.hindsight.vectorize.io"
_DEFAULT_LOCAL_URL = "http://localhost:8888"
_MIN_CLIENT_VERSION = "0.4.22"
_DEFAULT_TIMEOUT = 120  # seconds — cloud API can take 30-40s per request
_DEFAULT_IDLE_TIMEOUT = 300  # seconds — Hindsight embedded daemon default
# Mirrors hindsight-integrations/openclaw — Hindsight 0.5.0 added
# `update_mode='append'` semantics on retain (vectorize-io/hindsight#932).
# Without it, reusing a stable session-scoped document_id silently
# overwrites prior turns server-side, so we keep the per-process
# unique document_id fallback for older APIs.
_MIN_VERSION_FOR_UPDATE_MODE_APPEND = "0.5.0"
_VALID_BUDGETS = {"low", "mid", "high"}
_PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-120b",
    "openrouter": "qwen/qwen3.5-9b",
    "minimax": "MiniMax-M2.7",
    "ollama": "gemma3:12b",
    "lmstudio": "local-model",
    "openai_compatible": "your-model-name",
}


def _parse_int_setting(value: Any, default: int) -> int:
    """Parse an integer config/env value, falling back on invalid input."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid integer Hindsight setting %r; using default %s", value, default)
        return default


def _check_local_runtime() -> tuple[bool, str | None]:
    """Return whether local embedded Hindsight imports cleanly.

    On older CPUs, importing the local Hindsight stack can raise a runtime
    error from NumPy before the daemon starts. Treat that as "unavailable"
    so Hermes can degrade gracefully instead of repeatedly trying to start
    a broken local memory backend.
    """
    try:
        importlib.import_module("hindsight")
        importlib.import_module("hindsight_embed.daemon_embed_manager")
        return True, None
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Hindsight API capability probe — mirrors hindsight-integrations/openclaw.
# ---------------------------------------------------------------------------

# Cache of API_URL -> bool (whether that API supports update_mode='append').
# Probed once per URL per process — every provider talking to the same API
# gets the same answer without re-hitting /version on each initialize().
_append_capability_cache: Dict[str, bool] = {}
_append_capability_lock = threading.Lock()


def _meets_minimum_version(actual: str | None, required: str) -> bool:
    """Return True if *actual* ≥ *required* (semver). False on missing/invalid."""
    if not actual:
        return False
    try:
        from packaging.version import Version
        return Version(actual) >= Version(required)
    except Exception:
        return False


def _fetch_hindsight_api_version(api_url: str, api_key: str | None = None,
                                 timeout: float = 5.0) -> str | None:
    """GET ``<api_url>/version`` and return the version string (or None on failure).

    Hindsight's `/version` endpoint returns ``{"version": "0.5.6", ...}``.
    Any failure (timeout, 404, malformed JSON, missing key) → None, which
    the caller treats as "legacy API, no update_mode support".
    """
    import urllib.error
    import urllib.request
    if not api_url:
        return None
    url = api_url.rstrip("/") + "/version"
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            payload = resp.read().decode("utf-8", errors="replace")
        data = json.loads(payload)
    except Exception as exc:
        logger.debug("Hindsight /version probe failed for %s: %s", url, exc)
        return None
    if not isinstance(data, dict):
        return None
    version = data.get("version") or data.get("api_version")
    return str(version) if version else None


def _check_api_supports_update_mode_append(api_url: str,
                                           api_key: str | None = None) -> bool:
    """Cached capability check for ``update_mode='append'`` on *api_url*.

    Probes once per URL per process. Returns False on any probe failure —
    that's the safe default: a per-process unique ``document_id`` and no
    ``update_mode`` keeps the resume-overwrite fix (#6654) intact.
    """
    if not api_url:
        return False
    with _append_capability_lock:
        if api_url in _append_capability_cache:
            return _append_capability_cache[api_url]
    version = _fetch_hindsight_api_version(api_url, api_key)
    supported = _meets_minimum_version(version, _MIN_VERSION_FOR_UPDATE_MODE_APPEND)
    with _append_capability_lock:
        # Re-check after acquiring the lock in case a concurrent probe filled it.
        cached = _append_capability_cache.get(api_url)
        if cached is None:
            _append_capability_cache[api_url] = supported
        else:
            supported = cached
    if not supported:
        logger.warning(
            "Hindsight API at %s reports version %r, older than %s. "
            "Falling back to per-process document_id — retains across "
            "processes/sessions create separate documents instead of "
            "appending to a session-scoped one. Upgrade Hindsight to "
            "%s+ to enable update_mode='append' deduplication.",
            api_url, version, _MIN_VERSION_FOR_UPDATE_MODE_APPEND,
            _MIN_VERSION_FOR_UPDATE_MODE_APPEND,
        )
    else:
        logger.debug("Hindsight API %s version %s supports update_mode='append'",
                     api_url, version)
    return supported


# ---------------------------------------------------------------------------
# Dedicated event loop for Hindsight async calls (one per process, reused).
# Avoids creating ephemeral loops that leak aiohttp sessions.
# ---------------------------------------------------------------------------

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()

# Sentinel pushed to the per-provider retain queue to wake the writer for a
# clean exit. A unique object so it can never collide with a real job.
_WRITER_SENTINEL = object()


def _get_loop() -> asyncio.AbstractEventLoop:
    """Return a long-lived event loop running on a background thread."""
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is not None and _loop.is_running():
            return _loop
        _loop = asyncio.new_event_loop()

        def _run():
            asyncio.set_event_loop(_loop)
            _loop.run_forever()

        _loop_thread = threading.Thread(target=_run, daemon=True, name="hindsight-loop")
        _loop_thread.start()
        return _loop


def _run_sync(coro, timeout: float = _DEFAULT_TIMEOUT):
    """Schedule *coro* on the shared loop and block until done."""
    from agent.async_utils import safe_schedule_threadsafe
    loop = _get_loop()
    future = safe_schedule_threadsafe(coro, loop)
    if future is None:
        raise RuntimeError("Hindsight loop unavailable")
    return future.result(timeout=timeout)


# ---------------------------------------------------------------------------
# Backward-compatible alias — instances use self._run_sync() instead.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

RETAIN_SCHEMA = {
    "name": "hindsight_retain",
    "description": (
        "Store information to long-term memory. Hindsight automatically "
        "extracts structured facts, resolves entities, and indexes for retrieval."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to store."},
            "context": {"type": "string", "description": "Short label (e.g. 'user preference', 'project decision')."},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional per-call tags to merge with configured default retain tags.",
            },
            "raw": {
                "type": "boolean",
                "description": "If true, also write full text to Obsidian vault and store summary+pointer in hindsight.",
            },
        },
        "required": ["content"],
    },
}

RECALL_SCHEMA = {
    "name": "hindsight_recall",
    "description": (
        "Search long-term memory. Returns memories ranked by relevance using "
        "semantic search, keyword matching, entity graph traversal, and reranking."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "tags": {
                "type": "array", "items": {"type": "string"},
                "description": "Filter results to memories matching these tags (e.g. [\"scene:stock\"]). Uses recall_tags_match mode from config.",
            },
        },
        "required": ["query"],
    },
}

REFLECT_SCHEMA = {
    "name": "hindsight_reflect",
    "description": (
        "Synthesize a reasoned answer from long-term memories. Unlike recall, "
        "this reasons across all stored memories to produce a coherent response."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The question to reflect on."},
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from profile-scoped path, legacy path, or env vars.

    Resolution order:
      1. $HERMES_HOME/hindsight/config.json  (profile-scoped)
      2. ~/.hindsight/config.json             (legacy, shared)
      3. Environment variables
    """
    from pathlib import Path

    # Profile-scoped path (preferred)
    profile_path = get_hermes_home() / "hindsight" / "config.json"
    if profile_path.exists():
        try:
            return json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Legacy shared path (backward compat)
    legacy_path = Path.home() / ".hindsight" / "config.json"
    if legacy_path.exists():
        try:
            return json.loads(legacy_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "mode": os.environ.get("HINDSIGHT_MODE", "cloud"),
        "apiKey": os.environ.get("HINDSIGHT_API_KEY", ""),
        "timeout": _parse_int_setting(os.environ.get("HINDSIGHT_TIMEOUT"), _DEFAULT_TIMEOUT),
        "idle_timeout": _parse_int_setting(os.environ.get("HINDSIGHT_IDLE_TIMEOUT"), _DEFAULT_IDLE_TIMEOUT),
        "retain_tags": os.environ.get("HINDSIGHT_RETAIN_TAGS", ""),
        "retain_source": os.environ.get("HINDSIGHT_RETAIN_SOURCE", ""),
        "retain_user_prefix": os.environ.get("HINDSIGHT_RETAIN_USER_PREFIX", "User"),
        "retain_assistant_prefix": os.environ.get("HINDSIGHT_RETAIN_ASSISTANT_PREFIX", "Assistant"),
        "banks": {
            "hermes": {
                "bankId": os.environ.get("HINDSIGHT_BANK_ID", "hermes"),
                "budget": os.environ.get("HINDSIGHT_BUDGET", "mid"),
                "enabled": True,
            }
        },
    }


def _normalize_retain_tags(value: Any) -> List[str]:
    """Normalize tag config/tool values to a deduplicated list of strings."""
    if value is None:
        return []

    raw_items: list[Any]
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                raw_items = parsed
            else:
                raw_items = text.split(",")
        else:
            raw_items = text.split(",")
    else:
        raw_items = [value]

    normalized = []
    seen = set()
    for item in raw_items:
        tag = str(item).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
    return normalized


def _utc_timestamp() -> str:
    """Return current UTC timestamp in ISO-8601 with milliseconds and Z suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _embedded_profile_name(config: dict[str, Any]) -> str:
    """Return the Hindsight embedded profile name for this Hermes config."""
    profile = config.get("profile", "hermes")
    return str(profile or "hermes")


def _load_simple_env(path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file, ignoring comments and blank lines."""
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _build_embedded_profile_env(config: dict[str, Any], *, llm_api_key: str | None = None) -> dict[str, str]:
    """Build the profile-scoped env file that standalone hindsight-embed consumes."""
    current_key = llm_api_key
    if current_key is None:
        current_key = (
            config.get("llmApiKey")
            or config.get("llm_api_key")
            or os.environ.get("HINDSIGHT_LLM_API_KEY", "")
        )

    current_provider = config.get("llm_provider", "")
    current_model = config.get("llm_model", "")
    current_base_url = config.get("llm_base_url") or os.environ.get("HINDSIGHT_API_LLM_BASE_URL", "")

    # The embedded daemon expects OpenAI wire format for these providers.
    daemon_provider = "openai" if current_provider in {"openai_compatible", "openrouter"} else current_provider

    env_values = {
        "HINDSIGHT_API_LLM_PROVIDER": str(daemon_provider),
        "HINDSIGHT_API_LLM_API_KEY": str(current_key or ""),
        "HINDSIGHT_API_LLM_MODEL": str(current_model),
        "HINDSIGHT_API_LOG_LEVEL": "info",
    }
    if current_base_url:
        env_values["HINDSIGHT_API_LLM_BASE_URL"] = str(current_base_url)

    idle_timeout = (
        config.get("idle_timeout")
        if config.get("idle_timeout") is not None
        else os.environ.get("HINDSIGHT_IDLE_TIMEOUT")
    )
    if idle_timeout is not None and idle_timeout != "":
        env_values["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] = str(
            _parse_int_setting(idle_timeout, _DEFAULT_IDLE_TIMEOUT)
        )
    return env_values


def _embedded_profile_env_path(config: dict[str, Any]):
    from pathlib import Path

    return Path.home() / ".hindsight" / "profiles" / f"{_embedded_profile_name(config)}.env"


def _materialize_embedded_profile_env(config: dict[str, Any], *, llm_api_key: str | None = None):
    """Write the profile-scoped env file that standalone hindsight-embed uses."""
    profile_env = _embedded_profile_env_path(config)
    profile_env.parent.mkdir(parents=True, exist_ok=True)
    env_values = _build_embedded_profile_env(config, llm_api_key=llm_api_key)
    profile_env.write_text(
        "".join(f"{key}={value}\n" for key, value in env_values.items()),
        encoding="utf-8",
    )
    return profile_env

def _sanitize_bank_segment(value: str) -> str:
    """Sanitize a bank_id_template placeholder value.

    Bank IDs should be safe for URL paths and filesystem use. Replaces any
    character that isn't alphanumeric, dash, or underscore with a dash, and
    collapses runs of dashes.
    """
    if not value:
        return ""
    out = []
    prev_dash = False
    for ch in str(value):
        if ch.isalnum() or ch == "-" or ch == "_":
            out.append(ch)
            prev_dash = False
        else:
            if not prev_dash:
                out.append("-")
                prev_dash = True
    return "".join(out).strip("-_")


def _resolve_bank_id_template(template: str, fallback: str, **placeholders: str) -> str:
    """Resolve a bank_id template string with the given placeholders.

    Supported placeholders (each is sanitized before substitution):
      {profile}   — active Hermes profile name (from agent_identity)
      {workspace} — Hermes workspace name (from agent_workspace)
      {platform}  — "cli", "telegram", "discord", etc.
      {user}      — platform user id (gateway sessions)
      {session}   — current session id

    Missing/empty placeholders are rendered as the empty string and then
    collapsed — e.g. ``hermes-{user}`` with no user becomes ``hermes``.

    If the template is empty, resolution falls back to *fallback*.
    Returns the sanitized bank id.
    """
    if not template:
        return fallback
    sanitized = {k: _sanitize_bank_segment(v) for k, v in placeholders.items()}
    try:
        rendered = template.format(**sanitized)
    except (KeyError, IndexError) as exc:
        logger.warning("Invalid bank_id_template %r: %s — using fallback %r",
                       template, exc, fallback)
        return fallback
    while "--" in rendered:
        rendered = rendered.replace("--", "-")
    while "__" in rendered:
        rendered = rendered.replace("__", "_")
    rendered = rendered.strip("-_")
    return rendered or fallback


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class HindsightMemoryProvider(MemoryProvider):
    """Hindsight long-term memory with knowledge graph and multi-strategy retrieval."""

    def __init__(self):
        self._config = None
        self._api_key = None
        self._api_url = _DEFAULT_API_URL
        self._bank_id = "hermes"
        self._budget = "mid"
        self._mode = "cloud"
        self._llm_base_url = ""
        self._memory_mode = "hybrid"  # "context", "tools", or "hybrid"
        self._prefetch_method = "recall"  # "recall" or "reflect"
        self._retain_tags: List[str] = []
        self._retain_source = ""
        self._retain_user_prefix = "User"
        self._retain_assistant_prefix = "Assistant"
        self._platform = ""
        self._user_id = ""
        self._user_name = ""
        self._chat_id = ""
        self._chat_name = ""
        self._chat_type = ""
        self._thread_id = ""
        self._agent_identity = ""
        self._agent_workspace = ""
        self._turn_index = 0
        self._client = None
        self._timeout = _DEFAULT_TIMEOUT
        self._idle_timeout = _DEFAULT_IDLE_TIMEOUT
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        # Single-writer model for retain. sync_turn() enqueues; the writer
        # thread drains sequentially. Avoids spawning ad-hoc threads that
        # can race the interpreter shutdown and emit "cannot schedule new
        # futures after interpreter shutdown" / "Unclosed client session".
        self._retain_queue: queue.Queue = queue.Queue()
        self._writer_thread: threading.Thread | None = None
        self._shutting_down = threading.Event()
        self._atexit_registered = False
        # Legacy alias — older tests/callers reference _sync_thread directly.
        # Points at _writer_thread once the writer is running.
        self._sync_thread = None
        self._session_id = ""
        self._parent_session_id = ""
        self._document_id = ""

        # Tags
        self._tags: list[str] | None = None
        self._recall_tags: list[str] | None = None
        self._recall_tags_match = "any"

        # Retain controls
        self._auto_retain = True
        self._retain_every_n_turns = 1
        self._retain_async = True
        self._retain_context = "conversation between Hermes Agent and the User"
        self._turn_counter = 0
        self._session_turns: list[str] = []  # accumulates ALL turns for the session

        # Recall controls
        self._auto_recall = True
        self._recall_max_tokens = 4096
        self._recall_types: list[str] | None = None
        self._recall_prompt_preamble = ""
        self._recall_max_input_chars = 800

        # Bank
        self._bank_mission = ""
        self._bank_retain_mission: str | None = None
        self._bank_id_template = ""

    @property
    def name(self) -> str:
        return "hindsight"

    def is_available(self) -> bool:
        try:
            cfg = _load_config()
            mode = cfg.get("mode", "cloud")
            if mode in {"local", "local_embedded"}:
                available, _ = _check_local_runtime()
                return available
            if mode == "local_external":
                return True
            has_key = bool(
                cfg.get("apiKey")
                or cfg.get("api_key")
                or os.environ.get("HINDSIGHT_API_KEY", "")
            )
            has_url = bool(cfg.get("api_url") or os.environ.get("HINDSIGHT_API_URL", ""))
            return has_key or has_url
        except Exception:
            return False

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/hindsight/config.json."""
        import json
        from pathlib import Path
        config_dir = Path(hermes_home) / "hindsight"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        from utils import atomic_json_write
        atomic_json_write(config_path, existing, mode=0o600)

    def post_setup(self, hermes_home: str, config: dict) -> None:
        """Custom setup wizard — installs only the deps needed for the selected mode."""
        import getpass
        import subprocess
        import shutil
        import sys
        from pathlib import Path

        from hermes_cli.config import save_config

        from hermes_cli.memory_setup import _curses_select

        print("\n  Configuring Hindsight memory:\n")

        existing_config = self._config if isinstance(self._config, dict) else _load_config()
        if not isinstance(existing_config, dict):
            existing_config = {}

        # Step 1: Mode selection
        mode_values = ["cloud", "local_embedded", "local_external"]
        mode_items = [
            ("Cloud", "Hindsight Cloud API (lightweight, just needs an API key)"),
            ("Local Embedded", "Run Hindsight locally (downloads ~200MB, needs LLM key)"),
            ("Local External", "Connect to an existing Hindsight instance"),
        ]
        existing_mode = existing_config.get("mode")
        mode_default_idx = mode_values.index(existing_mode) if existing_mode in mode_values else 0
        mode_idx = _curses_select("  Select mode", mode_items, default=mode_default_idx)
        mode = mode_values[mode_idx]

        provider_config: dict = dict(existing_config)
        provider_config["mode"] = mode
        env_writes: dict = {}

        # Step 2: Install/upgrade deps for selected mode
        _MIN_CLIENT_VERSION = "0.4.22"
        cloud_dep = f"hindsight-client>={_MIN_CLIENT_VERSION}"
        local_dep = "hindsight-all"
        if mode == "local_embedded":
            deps_to_install = [local_dep]
        elif mode == "local_external":
            deps_to_install = [cloud_dep]
        else:
            deps_to_install = [cloud_dep]

        print("\n  Checking dependencies...")
        uv_path = shutil.which("uv")
        if not uv_path:
            print("  ⚠ uv not found — install it: curl -LsSf https://astral.sh/uv/install.sh | sh")
            print(f"  Then run manually: uv pip install --python {sys.executable} {' '.join(deps_to_install)}")
        else:
            try:
                subprocess.run(
                    [uv_path, "pip", "install", "--python", sys.executable, "--quiet", "--upgrade"] + deps_to_install,
                    check=True, timeout=120, capture_output=True,
                )
                print("  ✓ Dependencies up to date")
            except Exception as e:
                print(f"  ⚠ Install failed: {e}")
                print(f"  Run manually: uv pip install --python {sys.executable} {' '.join(deps_to_install)}")

        # Step 3: Mode-specific config
        if mode == "cloud":
            print("\n  Get your API key at https://ui.hindsight.vectorize.io\n")
            existing_key = os.environ.get("HINDSIGHT_API_KEY", "")
            if existing_key:
                masked = f"...{existing_key[-4:]}" if len(existing_key) > 4 else "set"
                sys.stdout.write(f"  API key (current: {masked}, blank to keep): ")
                sys.stdout.flush()
                api_key = getpass.getpass(prompt="") if sys.stdin.isatty() else sys.stdin.readline().strip()
            else:
                sys.stdout.write("  API key: ")
                sys.stdout.flush()
                api_key = getpass.getpass(prompt="") if sys.stdin.isatty() else sys.stdin.readline().strip()
            if api_key:
                env_writes["HINDSIGHT_API_KEY"] = api_key

            val = input(f"  API URL [{_DEFAULT_API_URL}]: ").strip()
            if val:
                provider_config["api_url"] = val

        elif mode == "local_external":
            val = input(f"  Hindsight API URL [{_DEFAULT_LOCAL_URL}]: ").strip()
            provider_config["api_url"] = val or _DEFAULT_LOCAL_URL

            sys.stdout.write("  API key (optional, blank to skip): ")
            sys.stdout.flush()
            api_key = getpass.getpass(prompt="") if sys.stdin.isatty() else sys.stdin.readline().strip()
            if api_key:
                env_writes["HINDSIGHT_API_KEY"] = api_key

        else:  # local_embedded
            providers_list = list(_PROVIDER_DEFAULT_MODELS.keys())
            llm_items = [
                (p, f"default model: {_PROVIDER_DEFAULT_MODELS[p]}")
                for p in providers_list
            ]
            existing_llm_provider = provider_config.get("llm_provider")
            llm_default_idx = providers_list.index(existing_llm_provider) if existing_llm_provider in providers_list else 0
            llm_idx = _curses_select("  Select LLM provider", llm_items, default=llm_default_idx)
            llm_provider = providers_list[llm_idx]

            provider_config["llm_provider"] = llm_provider

            if llm_provider == "openai_compatible":
                existing_base_url = provider_config.get("llm_base_url", "")
                prompt = "  LLM endpoint URL (e.g. http://192.168.1.10:8080/v1)"
                if existing_base_url:
                    prompt += f" [{existing_base_url}]"
                prompt += ": "
                val = input(prompt).strip()
                if val:
                    provider_config["llm_base_url"] = val
            elif llm_provider == "openrouter":
                provider_config["llm_base_url"] = "https://openrouter.ai/api/v1"

            provider_default_model = _PROVIDER_DEFAULT_MODELS.get(llm_provider, "gpt-4o-mini")
            current_model = provider_config.get("llm_model") or provider_default_model
            val = input(f"  LLM model [{current_model}]: ").strip()
            provider_config["llm_model"] = val or current_model

            sys.stdout.write("  LLM API key: ")
            sys.stdout.flush()
            llm_key = getpass.getpass(prompt="") if sys.stdin.isatty() else sys.stdin.readline().strip()
            if llm_key:
                env_writes["HINDSIGHT_LLM_API_KEY"] = llm_key
            else:
                env_path = Path(hermes_home) / ".env"
                existing_llm_key = ""
                if env_path.exists():
                    for line in env_path.read_text().splitlines():
                        if line.startswith("HINDSIGHT_LLM_API_KEY="):
                            existing_llm_key = line.split("=", 1)[1]
                            break
                env_writes["HINDSIGHT_LLM_API_KEY"] = existing_llm_key

        # Step 4: Save everything
        provider_config.setdefault("bank_id", "hermes")
        provider_config.setdefault("recall_budget", "mid")
        # Read existing timeout from config if present, otherwise use default.
        # Preserve explicit 0 values instead of treating them as blank.
        existing_timeout = provider_config.get("timeout")
        timeout_val = existing_timeout if existing_timeout is not None else _DEFAULT_TIMEOUT
        provider_config["timeout"] = timeout_val
        env_writes["HINDSIGHT_TIMEOUT"] = str(timeout_val)
        if mode == "local_embedded":
            existing_idle_timeout = provider_config.get("idle_timeout")
            idle_timeout_val = existing_idle_timeout if existing_idle_timeout is not None else _DEFAULT_IDLE_TIMEOUT
            provider_config["idle_timeout"] = idle_timeout_val
            env_writes["HINDSIGHT_IDLE_TIMEOUT"] = str(idle_timeout_val)
        config["memory"]["provider"] = "hindsight"
        save_config(config)

        self.save_config(provider_config, hermes_home)

        if env_writes:
            env_path = Path(hermes_home) / ".env"
            env_path.parent.mkdir(parents=True, exist_ok=True)
            existing_lines = []
            if env_path.exists():
                existing_lines = env_path.read_text().splitlines()
            updated_keys = set()
            new_lines = []
            for line in existing_lines:
                key_match = line.split("=", 1)[0].strip() if "=" in line and not line.startswith("#") else None
                if key_match and key_match in env_writes:
                    new_lines.append(f"{key_match}={env_writes[key_match]}")
                    updated_keys.add(key_match)
                else:
                    new_lines.append(line)
            for k, v in env_writes.items():
                if k not in updated_keys:
                    new_lines.append(f"{k}={v}")
            env_path.write_text("\n".join(new_lines) + "\n")

        if mode == "local_embedded":
            materialized_config = dict(provider_config)
            config_path = Path(hermes_home) / "hindsight" / "config.json"
            try:
                materialized_config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass

            llm_api_key = env_writes.get("HINDSIGHT_LLM_API_KEY", "")
            if not llm_api_key:
                llm_api_key = _load_simple_env(Path(hermes_home) / ".env").get("HINDSIGHT_LLM_API_KEY", "")
            if not llm_api_key:
                llm_api_key = _load_simple_env(_embedded_profile_env_path(materialized_config)).get(
                    "HINDSIGHT_API_LLM_API_KEY",
                    "",
                )

            _materialize_embedded_profile_env(
                materialized_config,
                llm_api_key=llm_api_key or None,
            )

        print(f"\n  ✓ Hindsight memory configured ({mode} mode)")
        if env_writes:
            print("  API keys saved to .env")
        print("\n  Start a new session to activate.\n")

    def get_config_schema(self):
        return [
            {"key": "mode", "description": "Connection mode", "default": "cloud", "choices": ["cloud", "local_embedded", "local_external"]},
            # Cloud mode
            {"key": "api_url", "description": "Hindsight Cloud API URL", "default": _DEFAULT_API_URL, "when": {"mode": "cloud"}},
            {"key": "api_key", "description": "Hindsight Cloud API key", "secret": True, "env_var": "HINDSIGHT_API_KEY", "url": "https://ui.hindsight.vectorize.io", "when": {"mode": "cloud"}},
            # Local external mode
            {"key": "api_url", "description": "Hindsight API URL", "default": _DEFAULT_LOCAL_URL, "when": {"mode": "local_external"}},
            {"key": "api_key", "description": "API key (optional)", "secret": True, "env_var": "HINDSIGHT_API_KEY", "when": {"mode": "local_external"}},
            # Local embedded mode
            {"key": "llm_provider", "description": "LLM provider", "default": "openai", "choices": ["openai", "anthropic", "gemini", "groq", "openrouter", "minimax", "ollama", "lmstudio", "openai_compatible"], "when": {"mode": "local_embedded"}},
            {"key": "llm_base_url", "description": "Endpoint URL (e.g. http://192.168.1.10:8080/v1)", "default": "", "when": {"mode": "local_embedded", "llm_provider": "openai_compatible"}},
            {"key": "llm_api_key", "description": "LLM API key (optional for openai_compatible)", "secret": True, "env_var": "HINDSIGHT_LLM_API_KEY", "when": {"mode": "local_embedded"}},
            {"key": "llm_model", "description": "LLM model", "default": "gpt-4o-mini", "default_from": {"field": "llm_provider", "map": _PROVIDER_DEFAULT_MODELS}, "when": {"mode": "local_embedded"}},
            {"key": "bank_id", "description": "Memory bank name (static fallback when bank_id_template is unset)", "default": "hermes"},
            {"key": "bank_id_template", "description": "Optional template to derive bank_id dynamically. Placeholders: {profile}, {workspace}, {platform}, {user}, {session}. Example: hermes-{profile}", "default": ""},
            {"key": "bank_mission", "description": "Mission/purpose description for the memory bank"},
            {"key": "bank_retain_mission", "description": "Custom extraction prompt for memory retention"},
            {"key": "recall_budget", "description": "Recall thoroughness", "default": "mid", "choices": ["low", "mid", "high"]},
            {"key": "memory_mode", "description": "Memory integration mode", "default": "hybrid", "choices": ["hybrid", "context", "tools"]},
            {"key": "recall_prefetch_method", "description": "Auto-recall method", "default": "recall", "choices": ["recall", "reflect"]},
            {"key": "retain_tags", "description": "Default tags applied to retained memories (comma-separated)", "default": ""},
            {"key": "retain_source", "description": "Metadata source value attached to retained memories", "default": ""},
            {"key": "retain_user_prefix", "description": "Label used before user turns in retained transcripts", "default": "User"},
            {"key": "retain_assistant_prefix", "description": "Label used before assistant turns in retained transcripts", "default": "Assistant"},
            {"key": "recall_tags", "description": "Tags to filter when searching memories (comma-separated)", "default": ""},
            {"key": "recall_tags_match", "description": "Tag matching mode for recall", "default": "any", "choices": ["any", "all", "any_strict", "all_strict"]},
            {"key": "auto_recall", "description": "Automatically recall memories before each turn", "default": True},
            {"key": "auto_retain", "description": "Automatically retain conversation turns", "default": True},
            {"key": "retain_every_n_turns", "description": "Retain every N turns (1 = every turn)", "default": 1},
            {"key": "retain_async","description": "Process retain asynchronously on the Hindsight server", "default": True},
            {"key": "retain_context", "description": "Context label for retained memories", "default": "conversation between Hermes Agent and the User"},
            {"key": "recall_max_tokens", "description": "Maximum tokens for recall results", "default": 4096},
            {"key": "recall_max_input_chars", "description": "Maximum input query length for auto-recall", "default": 800},
            {"key": "recall_prompt_preamble", "description": "Custom preamble for recalled memories in context"},
            {"key": "timeout", "description": "API request timeout in seconds", "default": _DEFAULT_TIMEOUT},
            {"key": "idle_timeout", "description": "Embedded daemon idle timeout in seconds (0 disables auto-shutdown)", "default": _DEFAULT_IDLE_TIMEOUT, "when": {"mode": "local_embedded"}},
        ]

    def _get_client(self):
        """Return the cached Hindsight client (created once, reused)."""
        if self._client is None:
            if self._mode == "local_embedded":
                available, reason = _check_local_runtime()
                if not available:
                    raise RuntimeError(
                        "Hindsight local runtime is unavailable"
                        + (f": {reason}" if reason else "")
                    )
                try:
                    from tools.lazy_deps import ensure as _lazy_ensure
                    _lazy_ensure("memory.hindsight", prompt=False)
                except ImportError:
                    pass
                except Exception as _e:
                    raise ImportError(str(_e))
                from hindsight import HindsightEmbedded
                HindsightEmbedded.__del__ = lambda self: None
                llm_provider = self._config.get("llm_provider", "")
                if llm_provider in {"openai_compatible", "openrouter"}:
                    llm_provider = "openai"
                logger.debug("Creating HindsightEmbedded client (profile=%s, provider=%s)",
                             self._config.get("profile", "hermes"), llm_provider)
                kwargs = dict(
                    profile=self._config.get("profile", "hermes"),
                    llm_provider=llm_provider,
                    llm_api_key=self._config.get("llmApiKey") or self._config.get("llm_api_key") or os.environ.get("HINDSIGHT_LLM_API_KEY", ""),
                    llm_model=self._config.get("llm_model", ""),
                )
                if self._llm_base_url:
                    kwargs["llm_base_url"] = self._llm_base_url
                idle_timeout = _parse_int_setting(
                    self._config.get("idle_timeout")
                    if self._config.get("idle_timeout") is not None
                    else os.environ.get("HINDSIGHT_IDLE_TIMEOUT", self._idle_timeout),
                    _DEFAULT_IDLE_TIMEOUT,
                )
                self._idle_timeout = idle_timeout
                kwargs["idle_timeout"] = idle_timeout
                self._client = HindsightEmbedded(**kwargs)
            else:
                from hindsight_client import Hindsight
                timeout = self._timeout or _DEFAULT_TIMEOUT
                kwargs = {"base_url": self._api_url, "timeout": float(timeout)}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                logger.debug("Creating Hindsight cloud client (url=%s, has_key=%s, timeout=%s)",
                             self._api_url, bool(self._api_key), kwargs["timeout"])
                self._client = Hindsight(**kwargs)
        return self._client

    def _run_sync(self, coro):
        """Schedule *coro* on the shared loop using the configured timeout."""
        return _run_sync(coro, timeout=self._timeout)

    def _is_retriable_embedded_connection_error(self, exc: Exception) -> bool:
        """Return True for stale embedded-daemon connection failures."""
        if self._mode != "local_embedded":
            return False
        text = f"{type(exc).__name__}: {exc}".lower()
        return any(
            marker in text
            for marker in (
                "cannot connect to host",
                "connection refused",
                "connect call failed",
                "clientconnectorerror",
            )
        )

    def _ensure_writer(self) -> None:
        """Lazy-start the single retain-writer thread.

        We don't start the writer in initialize() so providers that never
        retain (e.g. tools-only mode) don't pay for an idle thread.
        """
        thread = self._writer_thread
        if thread is not None and thread.is_alive():
            return
        # If the previous writer exited (e.g. after a prior shutdown), reset
        # the flag so this fresh writer is allowed to drain new jobs.
        self._shutting_down.clear()
        thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="hindsight-writer",
        )
        self._writer_thread = thread
        # Keep the legacy _sync_thread alias pointing at the writer so any
        # external code that joins _sync_thread keeps working.
        self._sync_thread = thread
        thread.start()

    def _writer_loop(self) -> None:
        """Drain the retain queue serially. Exits on sentinel.

        Each job() is wrapped so a single failure can't kill the writer.
        task_done() always fires so queue.join() works in tests.
        """
        while True:
            try:
                job = self._retain_queue.get(timeout=1.0)
            except queue.Empty:
                if self._shutting_down.is_set():
                    return
                continue
            try:
                if job is _WRITER_SENTINEL:
                    return
                try:
                    job()
                except Exception as exc:
                    logger.warning("Hindsight retain failed: %s", exc, exc_info=True)
            finally:
                self._retain_queue.task_done()

    def _register_atexit(self) -> None:
        """Register an idempotent atexit hook to drain the writer.

        Without this, a CLI exit that doesn't go through MemoryManager.
        shutdown_all() would leave in-flight retain jobs racing interpreter
        teardown, producing "cannot schedule new futures" warnings and
        unclosed aiohttp sessions.
        """
        if self._atexit_registered:
            return
        self._atexit_registered = True
        atexit.register(self._atexit_shutdown)

    def _atexit_shutdown(self) -> None:
        if self._shutting_down.is_set():
            return
        try:
            self.shutdown()
        except Exception as exc:
            logger.debug("Hindsight atexit shutdown failed: %s", exc)

    def _run_hindsight_operation(self, operation):
        """Run an async Hindsight client operation, retrying once after idle shutdown."""
        client = self._get_client()
        try:
            return self._run_sync(operation(client))
        except Exception as exc:
            if not self._is_retriable_embedded_connection_error(exc):
                raise
            logger.info(
                "Hindsight embedded daemon appears unreachable; recreating client and retrying once: %s",
                exc,
            )
            self._client = None
            client = self._get_client()
            self._client = client
            return self._run_sync(operation(client))

    def _probe_url(self) -> str:
        """Return the URL to probe /version on.

        For local_embedded the daemon is on a per-profile dynamic port,
        so we prefer the running client's URL when available; otherwise
        fall back to the configured api_url.
        """
        if self._mode == "local_embedded" and self._client is not None:
            url = getattr(self._client, "url", None)
            if url:
                return str(url)
        return self._api_url or ""

    def _resolve_retain_target(self, fallback_document_id: str) -> tuple[str, str | None]:
        """Pick (document_id, update_mode) based on live API capability.

        On Hindsight ≥ 0.5.0 the API supports ``update_mode='append'``,
        which lets us reuse a stable session-scoped ``document_id`` across
        process lifecycles without overwriting prior turns. On older APIs
        we fall back to *fallback_document_id* (the per-process unique
        ``f"{session_id}-{start_ts}"`` minted at initialize / switch time)
        and don't pass ``update_mode`` at all — that's the only way the
        resume-overwrite fix (#6654) keeps working on legacy servers.

        Probe is cached at module level per API URL, so this is one HTTP
        round-trip per (process, api_url) pair regardless of how many
        retains fire.
        """
        if not self._session_id:
            return fallback_document_id, None
        if _check_api_supports_update_mode_append(self._probe_url(), self._api_key):
            return self._session_id, "append"
        return fallback_document_id, None

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = str(session_id or "").strip()
        self._parent_session_id = str(kwargs.get("parent_session_id", "") or "").strip()

        # Each process lifecycle gets its own document_id. Reusing session_id
        # alone caused overwrites on /resume — the reloaded session starts
        # with an empty _session_turns, so the next retain would replace the
        # previously stored content. session_id stays in tags so processes
        # for the same session remain filterable together.
        start_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._document_id = f"{self._session_id}-{start_ts}"

        # Check client version and auto-upgrade if needed
        try:
            from importlib.metadata import version as pkg_version
            from packaging.version import Version
            installed = pkg_version("hindsight-client")
            if Version(installed) < Version(_MIN_CLIENT_VERSION):
                logger.warning("hindsight-client %s is outdated (need >=%s), attempting upgrade...",
                               installed, _MIN_CLIENT_VERSION)
                import shutil
                import subprocess
                import sys
                uv_path = shutil.which("uv")
                if uv_path:
                    try:
                        subprocess.run(
                            [uv_path, "pip", "install", "--python", sys.executable,
                             "--quiet", "--upgrade", f"hindsight-client>={_MIN_CLIENT_VERSION}"],
                            check=True, timeout=120, capture_output=True,
                        )
                        logger.info("hindsight-client upgraded to >=%s", _MIN_CLIENT_VERSION)
                    except Exception as e:
                        logger.warning("Auto-upgrade failed: %s. Run: uv pip install 'hindsight-client>=%s'",
                                       e, _MIN_CLIENT_VERSION)
                else:
                    logger.warning("uv not found. Run: pip install 'hindsight-client>=%s'", _MIN_CLIENT_VERSION)
        except Exception:
            pass  # packaging not available or other issue — proceed anyway

        self._config = _load_config()
        self._platform = str(kwargs.get("platform") or "").strip()
        self._user_id = str(kwargs.get("user_id") or "").strip()
        self._user_name = str(kwargs.get("user_name") or "").strip()
        self._chat_id = str(kwargs.get("chat_id") or "").strip()
        self._chat_name = str(kwargs.get("chat_name") or "").strip()
        self._chat_type = str(kwargs.get("chat_type") or "").strip()
        self._thread_id = str(kwargs.get("thread_id") or "").strip()
        self._agent_identity = str(kwargs.get("agent_identity") or "").strip()
        self._agent_workspace = str(kwargs.get("agent_workspace") or "").strip()
        self._turn_index = 0
        self._session_turns = []
        self._mode = self._config.get("mode", "cloud")
        # Read timeout from config or env var, fall back to default
        self._timeout = _parse_int_setting(
            self._config.get("timeout") if self._config.get("timeout") is not None else os.environ.get("HINDSIGHT_TIMEOUT"),
            _DEFAULT_TIMEOUT,
        )
        self._idle_timeout = _parse_int_setting(
            self._config.get("idle_timeout") if self._config.get("idle_timeout") is not None else os.environ.get("HINDSIGHT_IDLE_TIMEOUT"),
            _DEFAULT_IDLE_TIMEOUT,
        )
        # "local" is a legacy alias for "local_embedded"
        if self._mode == "local":
            self._mode = "local_embedded"
        if self._mode == "local_embedded":
            available, reason = _check_local_runtime()
            if not available:
                logger.warning(
                    "Hindsight local mode disabled because its runtime could not be imported: %s",
                    reason,
                )
                self._mode = "disabled"
                return
        self._api_key = self._config.get("apiKey") or self._config.get("api_key") or os.environ.get("HINDSIGHT_API_KEY", "")
        default_url = _DEFAULT_LOCAL_URL if self._mode in {"local_embedded", "local_external"} else _DEFAULT_API_URL
        self._api_url = self._config.get("api_url") or os.environ.get("HINDSIGHT_API_URL", default_url)
        self._llm_base_url = self._config.get("llm_base_url", "")

        banks = cfg_get(self._config, "banks", "hermes", default={})
        static_bank_id = self._config.get("bank_id") or banks.get("bankId", "hermes")
        self._bank_id_template = self._config.get("bank_id_template", "") or ""
        self._bank_id = _resolve_bank_id_template(
            self._bank_id_template,
            fallback=static_bank_id,
            profile=self._agent_identity,
            workspace=self._agent_workspace,
            platform=self._platform,
            user=self._user_id,
            session=self._session_id,
        )
        budget = self._config.get("recall_budget") or self._config.get("budget") or banks.get("budget", "mid")
        self._budget = budget if budget in _VALID_BUDGETS else "mid"

        memory_mode = self._config.get("memory_mode", "hybrid")
        self._memory_mode = memory_mode if memory_mode in {"context", "tools", "hybrid"} else "hybrid"

        prefetch_method = self._config.get("recall_prefetch_method") or self._config.get("prefetch_method", "recall")
        self._prefetch_method = prefetch_method if prefetch_method in {"recall", "reflect"} else "recall"

        # Bank options
        self._bank_mission = self._config.get("bank_mission", "")
        self._bank_retain_mission = self._config.get("bank_retain_mission") or None

        # Tags
        self._retain_tags = _normalize_retain_tags(
            self._config.get("retain_tags")
            or os.environ.get("HINDSIGHT_RETAIN_TAGS", "")
        )
        self._tags = self._retain_tags or None
        self._recall_tags = self._config.get("recall_tags") or None
        self._recall_tags_match = self._config.get("recall_tags_match", "any")
        self._retain_source = str(
            self._config.get("retain_source") or os.environ.get("HINDSIGHT_RETAIN_SOURCE", "")
        ).strip()
        self._retain_user_prefix = str(
            self._config.get("retain_user_prefix") or os.environ.get("HINDSIGHT_RETAIN_USER_PREFIX", "User")
        ).strip() or "User"
        self._retain_assistant_prefix = str(
            self._config.get("retain_assistant_prefix") or os.environ.get("HINDSIGHT_RETAIN_ASSISTANT_PREFIX", "Assistant")
        ).strip() or "Assistant"

        # Retain controls
        self._auto_retain = self._config.get("auto_retain", True)
        self._retain_every_n_turns = max(1, int(self._config.get("retain_every_n_turns", 1)))
        self._retain_context = self._config.get("retain_context", "conversation between Hermes Agent and the User")

        # Recall controls
        self._auto_recall = self._config.get("auto_recall", True)
        self._recall_max_tokens = int(self._config.get("recall_max_tokens", 4096))
        self._recall_types = self._config.get("recall_types") or None
        self._recall_prompt_preamble = self._config.get("recall_prompt_preamble", "")
        self._recall_max_input_chars = int(self._config.get("recall_max_input_chars", 800))
        self._retain_async = self._config.get("retain_async", True)

        _client_version = "unknown"
        try:
            from importlib.metadata import version as pkg_version
            _client_version = pkg_version("hindsight-client")
        except Exception:
            pass
        logger.info("Hindsight initialized: mode=%s, api_url=%s, bank=%s, budget=%s, memory_mode=%s, prefetch_method=%s, client=%s",
                     self._mode, self._api_url, self._bank_id, self._budget, self._memory_mode, self._prefetch_method, _client_version)
        if self._bank_id_template:
            logger.debug("Hindsight bank resolved from template %r: profile=%s workspace=%s platform=%s user=%s -> bank=%s",
                         self._bank_id_template, self._agent_identity, self._agent_workspace,
                         self._platform, self._user_id, self._bank_id)
        logger.debug("Hindsight config: auto_retain=%s, auto_recall=%s, retain_every_n=%d, "
                     "retain_async=%s, retain_context=%s, recall_max_tokens=%d, recall_max_input_chars=%d, tags=%s, recall_tags=%s",
                     self._auto_retain, self._auto_recall, self._retain_every_n_turns,
                     self._retain_async, self._retain_context, self._recall_max_tokens, self._recall_max_input_chars,
                     self._tags, self._recall_tags)

        # For local mode, start the embedded daemon in the background so it
        # doesn't block the chat. Redirect stdout/stderr to a log file to
        # prevent rich startup output from spamming the terminal.
        if self._mode == "local_embedded":
            def _start_daemon():
                import traceback
                log_dir = get_hermes_home() / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / "hindsight-embed.log"
                try:
                    # Redirect the daemon manager's Rich console to our log file
                    # instead of stderr. This avoids global fd redirects that
                    # would capture output from other threads.
                    import hindsight_embed.daemon_embed_manager as dem
                    from rich.console import Console
                    dem.console = Console(file=open(log_path, "a", encoding="utf-8"), force_terminal=False)

                    client = self._get_client()
                    profile = self._config.get("profile", "hermes")

                    # Update the profile .env to match our current config so
                    # the daemon always starts with the right settings.
                    # If the config changed and the daemon is running, stop it.
                    profile_env = _embedded_profile_env_path(self._config)
                    expected_env = _build_embedded_profile_env(self._config)
                    saved = _load_simple_env(profile_env)
                    config_changed = saved != expected_env

                    if config_changed:
                        profile_env = _materialize_embedded_profile_env(self._config)
                        if client._manager.is_running(profile):
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write("\n=== Config changed, restarting daemon ===\n")
                            client._manager.stop(profile)

                    client._ensure_started()
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write("\n=== Daemon started successfully ===\n")
                except Exception as e:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"\n=== Daemon startup failed: {e} ===\n")
                        traceback.print_exc(file=f)

            t = threading.Thread(target=_start_daemon, daemon=True, name="hindsight-daemon-start")
            t.start()

    def system_prompt_block(self) -> str:
        if self._memory_mode == "context":
            return (
                f"# Hindsight Memory\n"
                f"Active (context mode). Bank: {self._bank_id}, budget: {self._budget}.\n"
                f"Relevant memories are automatically injected into context."
            )
        if self._memory_mode == "tools":
            return (
                f"# Hindsight Memory\n"
                f"Active (tools mode). Bank: {self._bank_id}, budget: {self._budget}.\n"
                f"Use hindsight_recall to search, hindsight_reflect for synthesis, "
                f"hindsight_retain to store facts."
            )
        return (
            f"# Hindsight Memory\n"
            f"Active. Bank: {self._bank_id}, budget: {self._budget}.\n"
            f"Relevant memories are automatically injected into context. "
            f"Use hindsight_recall to search, hindsight_reflect for synthesis, "
            f"hindsight_retain to store facts."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            logger.debug("Prefetch: waiting for background thread to complete")
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            logger.debug("Prefetch: no results available")
            return ""
        logger.debug("Prefetch: returning %d chars of context", len(result))
        header = self._recall_prompt_preamble or (
            "# Hindsight Memory (persistent cross-session context)\n"
            "Use this to answer questions about the user and prior sessions. "
            "Do not call tools to look up information that is already present here."
        )
        return f"{header}\n\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._memory_mode == "tools":
            logger.debug("Prefetch: skipped (tools-only mode)")
            return
        if not self._auto_recall:
            logger.debug("Prefetch: skipped (auto_recall disabled)")
            return
        if self._shutting_down.is_set():
            logger.debug("Prefetch: skipped (shutting down)")
            return
        # Truncate query to max chars
        if self._recall_max_input_chars and len(query) > self._recall_max_input_chars:
            query = query[:self._recall_max_input_chars]

        def _run():
            try:
                if self._prefetch_method == "reflect":
                    logger.debug("Prefetch: calling reflect (bank=%s, query_len=%d)", self._bank_id, len(query))
                    resp = self._run_hindsight_operation(lambda client: client.areflect(bank_id=self._bank_id, query=query, budget=self._budget))
                    text = resp.text or ""
                else:
                    recall_kwargs: dict = {
                        "bank_id": self._bank_id, "query": query,
                        "budget": self._budget, "max_tokens": self._recall_max_tokens,
                    }
                    if self._recall_tags:
                        recall_kwargs["tags"] = self._recall_tags
                        recall_kwargs["tags_match"] = self._recall_tags_match
                    if self._recall_types:
                        recall_kwargs["types"] = self._recall_types
                    logger.debug("Prefetch: calling recall (bank=%s, query_len=%d, budget=%s)",
                                 self._bank_id, len(query), self._budget)
                    resp = self._run_hindsight_operation(lambda client: client.arecall(**recall_kwargs))
                    num_results = len(resp.results) if resp.results else 0
                    logger.debug("Prefetch: recall returned %d results", num_results)
                    # 艾宾浩斯时间衰减重排序
                    scored = self._rerank_with_time_decay(resp.results or [])
                    # 去重：daily-summary 重复记忆，保留分数最高版本
                    scored = self._dedup_results(scored)
                    # 场景隔离重排序：检测查询场景，同场景结果优先
                    scored = self._scene_filter_results(scored, query)
                    # ── P0/P1/P2/P3 分级过滤 ──
                    # P0: entity|object 标签 → 全量保留（高价值，通常带 AMAP 实体映射）
                    # P1: 纯 auto_retain（无 entity|object）→ 最多 3 条（备查即可）
                    # P2: 其它（非 skill 非 auto）→ 正常保留
                    # P3: skill 日常更新类 → 压缩为摘要，保留实体+动作+时间
                    try:
                        import re
                        p0, p1_auto, p2_other, p3 = [], [], [], []
                        for s in scored:
                            content = s[3] if len(s) > 3 else ""
                            tags = s[4] if len(s) > 4 else []
                            has_entity_obj = any(str(t).startswith("entity|object:") for t in tags)
                            has_auto_retain = any("entity|concept:auto_retain" in str(t) for t in tags)
                            is_skill = any(kw in content.lower() for kw in
                                           ["skill update", "skill library", "skills 更新"])
                            if has_entity_obj:
                                p0.append(s)      # P0: 高价值
                            elif has_auto_retain and not has_entity_obj:
                                p1_auto.append(s)  # P1: 纯 auto_retain，限流
                            elif is_skill:
                                p3.append(s)       # P3: skill 更新
                            else:
                                p2_other.append(s) # P2: 其他有价值内容
                        # P1: auto_retain 最多保留 3 条
                        auto_cap = min(len(p1_auto), 3)
                        if len(p1_auto) > 3:
                            logger.debug("Prefetch P1(auto_retain): capped %d→3", len(p1_auto))
                        p1_capped = p1_auto[:3]
                        # 压缩 P3：提取核心事实，不丢 who/when
                        if p3:
                            old_count = len(p3)
                            seen_fingerprint = set()
                            p3_compressed = []
                            for s in p3:
                                content = s[3]
                                # 提取核心事实：取 "| When:" 之前的部分作为key
                                # 格式: "<实体> <动作> <详情> | When: <日期> | Involving: <谁>"
                                core_part = content.split("| When:")[0].split("| 时间:")[0].strip()
                                # 规范化fingerprint用于去重
                                fp = re.sub(r'\s+', ' ', core_part.lower())[:50]
                                if fp not in seen_fingerprint:
                                    seen_fingerprint.add(fp)
                                    # 核心事实完整保留，不需要截断
                                    p3_compressed.append(
                                        (s[0], s[1] * 0.6, s[2], core_part,
                                         s[4] if len(s) > 4 else [])
                                    )
                            logger.debug("Prefetch P0/P1/P2/P3: P0=%d P1=%d→%d P2=%d P3=%d→%d",
                                         len(p0), len(p1_auto), len(p1_capped),
                                         len(p2_other), old_count, len(p3_compressed))
                            scored = p0 + p1_capped + p2_other + p3_compressed
                        else:
                            logger.debug("Prefetch P0/P1/P2/P3: P0=%d P1=%d→%d P2=%d P3=0",
                                         len(p0), len(p1_auto), len(p1_capped), len(p2_other))
                    except Exception as e:
                        logger.debug("Prefetch P0/P1/P2/P3 grading failed, using raw results: %s", e)

                    # ── 跨语言实体去重 ──
                    # 中英文描述同一事实时保留分数更高的版本
                    # 指纹提取：取英文技术标识符(含下划线/大写/数字) + 中文短语辅助
                    try:
                        import re as _re
                        _ENG_STOP = {'the','was','were','this','that','with','from',
                            'been','have','will','would','could','should','their',
                            'there','which','also','after','into','than','over',
                            'about','said','each','when','what','more','some',
                            'them','then','very','just','because','before','have'}
                        deduped_final = []
                        seen_entity = set()
                        for s in scored:
                            content = s[3]
                            core = _re.sub(r'(?i)\|\s*(when|involving|时间)\s*[:：].*', '', content)
                            # 只保留含特殊字符/大写/数字的英文词（技术标识符特征）
                            eng_terms = _re.findall(r'[a-zA-Z_][a-zA-Z0-9_-]{3,}', core)
                            tech_terms = sorted(set(
                                t.lower() for t in eng_terms
                                if _re.search(r'[_-]|[A-Z]|\d', t)
                                and t.lower() not in _ENG_STOP
                            ))[:3]
                            cn_phrases = _re.findall(r'[\u4e00-\u9fff]{2,}', core)
                            if tech_terms:
                                entity_fp = "::".join(tech_terms)
                            elif cn_phrases:
                                entity_fp = cn_phrases[0][:8]
                            else:
                                entity_fp = core[:30].lower().strip()
                            if entity_fp not in seen_entity:
                                seen_entity.add(entity_fp)
                                deduped_final.append(s)
                        removed_x = len(scored) - len(deduped_final)
                        if removed_x > 0:
                            logger.debug("Prefetch cross-lang dedup: %d→%d", len(scored), len(deduped_final))
                        scored = deduped_final
                    except Exception:
                        pass

                    # 格式化上下文只带 scene 标签，区分场景噪声最小
                    lines = []
                    for s in scored:
                        content = s[3] if len(s) > 3 else ""
                        tags = s[4] if len(s) > 4 and s[4] else []
                        scene_tags = [t for t in tags if t.startswith("scene:")]
                        tag_str = f" [{','.join(scene_tags[:3])}]" if scene_tags else ""
                        lines.append(f"- {content}{tag_str}")
                    text = "\n".join(lines) if lines else ""

                # ── PPR 图扩散扩展 ──
                if self._prefetch_method != "reflect":
                    try:
                        # 每10次 prefetch 刷新一次动态路由
                        self._AMAP_DYNAMIC_PREFETCH_COUNT += 1
                        if self._AMAP_DYNAMIC_PREFETCH_COUNT % 10 == 0:
                            self._update_amap_routes()

                        # Build seed nodes from AMAP + recall results
                        seed_nodes = []
                        amap_entity = self._try_amap_match(query)
                        if amap_entity:
                            seed_nodes.append(amap_entity)

                        # 从语义召回结果中提取 entity 标签作为种子
                        if resp and resp.results:
                            for r in resp.results[:5]:
                                for tag in (getattr(r, 'tags', None) or []):
                                    ts = str(tag)
                                    if ts.startswith("entity|"):
                                        if ts not in seed_nodes:
                                            seed_nodes.append(ts)

                        if seed_nodes:
                            # 构建/刷新图
                            n_nodes = self._memory_graph.build_from_hindsight()
                            logger.debug("Prefetch PPR: graph has %d nodes, seeds=%s",
                                         n_nodes, seed_nodes[:3])

                            # PPR 扩散
                            expanded = self._memory_graph.get_expanded_nodes(
                                seed_nodes, top_k=5, min_score=0.005)

                            if expanded:
                                extra_lines = []
                                for node, score in expanded:
                                    tag_results = self._tag_recall(node, limit=2)
                                    for r in tag_results:
                                        if r.text:
                                            extra_lines.append(
                                                f"- [关联:{score:.2f}] {r.text}")
                                if extra_lines:
                                    extra_text = "\n".join(extra_lines[:8])
                                    if text:
                                        text = text + "\n# 联想记忆扩展\n" + extra_text
                                    else:
                                        text = "# 联想记忆扩展\n" + extra_text
                                    logger.debug("Prefetch PPR: added %d expanded memories",
                                                 len(extra_lines))
                    except Exception as ppr_e:
                        logger.debug("Prefetch PPR expand failed: %s", ppr_e, exc_info=True)

                if text:
                    with self._prefetch_lock:
                        self._prefetch_result = text
            except Exception as e:
                logger.debug("Hindsight prefetch failed: %s", e, exc_info=True)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="hindsight-prefetch")
        self._prefetch_thread.start()

    def _build_turn_messages(self, user_content: str, assistant_content: str) -> List[Dict[str, str]]:
        now = datetime.now(timezone.utc).isoformat()
        return [
            {
                "role": "user",
                "content": f"{self._retain_user_prefix}: {user_content}",
                "timestamp": now,
            },
            {
                "role": "assistant",
                "content": f"{self._retain_assistant_prefix}: {assistant_content}",
                "timestamp": now,
            },
        ]

    def _build_metadata(self, *, message_count: int, turn_index: int) -> Dict[str, str]:
        metadata: Dict[str, str] = {
            "retained_at": _utc_timestamp(),
            "message_count": str(message_count),
            "turn_index": str(turn_index),
        }
        if self._retain_source:
            metadata["source"] = self._retain_source
        if self._session_id:
            metadata["session_id"] = self._session_id
        if self._platform:
            metadata["platform"] = self._platform
        if self._user_id:
            metadata["user_id"] = self._user_id
        if self._user_name:
            metadata["user_name"] = self._user_name
        if self._chat_id:
            metadata["chat_id"] = self._chat_id
        if self._chat_name:
            metadata["chat_name"] = self._chat_name
        if self._chat_type:
            metadata["chat_type"] = self._chat_type
        if self._thread_id:
            metadata["thread_id"] = self._thread_id
        if self._agent_identity:
            metadata["agent_identity"] = self._agent_identity
        return metadata

    def _build_retain_kwargs(
        self,
        content: str,
        *,
        context: str | None = None,
        document_id: str | None = None,
        metadata: Dict[str, str] | None = None,
        tags: List[str] | None = None,
        retain_async: bool | None = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "bank_id": self._bank_id,
            "content": content,
            "metadata": metadata or self._build_metadata(message_count=1, turn_index=self._turn_index),
        }
        if context is not None:
            kwargs["context"] = context
        if document_id:
            kwargs["document_id"] = document_id
        if retain_async is not None:
            kwargs["retain_async"] = retain_async
        merged_tags = _normalize_retain_tags(self._retain_tags)
        for tag in _normalize_retain_tags(tags):
            if tag not in merged_tags:
                merged_tags.append(tag)
        if merged_tags:
            kwargs["tags"] = merged_tags
        return kwargs

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Enqueue a retain for the current turn. Non-blocking.

        The actual aretain_batch runs on a single long-lived writer thread
        that drains an in-memory queue. Once shutdown() has been called,
        further sync_turn() calls are dropped — this prevents post-exit
        retains from reaching aiohttp after interpreter shutdown begins.
        """
        if not self._auto_retain:
            logger.debug("sync_turn: skipped (auto_retain disabled)")
            return
        if self._shutting_down.is_set():
            logger.debug("sync_turn: skipped (shutting down)")
            return

        if session_id:
            self._session_id = str(session_id).strip()

        turn = json.dumps(self._build_turn_messages(user_content, assistant_content), ensure_ascii=False)
        self._session_turns.append(turn)
        self._turn_counter += 1
        self._turn_index = self._turn_counter

        if self._turn_counter % self._retain_every_n_turns != 0:
            logger.debug("sync_turn: buffered turn %d (will retain at turn %d)",
                         self._turn_counter, self._turn_counter + (self._retain_every_n_turns - self._turn_counter % self._retain_every_n_turns))
            return

        logger.debug("sync_turn: retaining %d turns, total session content %d chars",
                     len(self._session_turns), sum(len(t) for t in self._session_turns))
        content = "[" + ",".join(self._session_turns) + "]"

        lineage_tags: list[str] = []
        if self._session_id:
            lineage_tags.append(f"session:{self._session_id}")
        if self._parent_session_id:
            lineage_tags.append(f"parent:{self._parent_session_id}")

        # Snapshot the state needed for the retain. The writer may run after
        # _session_turns / _turn_index are mutated by a later sync_turn().
        metadata_snapshot = self._build_metadata(
            message_count=len(self._session_turns) * 2,
            turn_index=self._turn_index,
        )
        num_turns = len(self._session_turns)
        document_id, update_mode = self._resolve_retain_target(self._document_id)
        bank_id = self._bank_id
        retain_async_flag = self._retain_async
        retain_context = self._retain_context

        def _do_retain() -> None:
            item = self._build_retain_kwargs(
                content,
                context=retain_context,
                metadata=metadata_snapshot,
                tags=lineage_tags or None,
            )
            item.pop("bank_id", None)
            item.pop("retain_async", None)
            if update_mode is not None:
                item["update_mode"] = update_mode
            logger.debug("Hindsight retain: bank=%s, doc=%s, mode=%s, async=%s, content_len=%d, num_turns=%d",
                         bank_id, document_id, update_mode, retain_async_flag, len(content), num_turns)
            self._run_hindsight_operation(
                lambda client: client.aretain_batch(
                    bank_id=bank_id,
                    items=[item],
                    document_id=document_id,
                    retain_async=retain_async_flag,
                )
            )
            logger.debug("Hindsight retain succeeded")

        self._ensure_writer()
        self._register_atexit()
        self._retain_queue.put(_do_retain)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if self._memory_mode == "context":
            return []
        return [RETAIN_SCHEMA, RECALL_SCHEMA, REFLECT_SCHEMA]

    # ── Scene/entity tag inference ──────────────────────────────────

    def _infer_scene(self, content: str) -> Optional[str]:
        """根据内容推断场景标签"""
        c = content.lower()
        scene_patterns = [
            ("scene:stock", ["股票", "股价", "持仓", "买入", "卖出", "止损", "加仓",
                            "k线", "boll", "adx", "市盈率", "pe", "pb",
                            "002", "300", "600", "688", "sz.", "sh."]),
            ("scene:dev", ["代码", "bug", "pr", "merge", "commit", "重构",
                          "部署", "config", "api", "函数", "模块", "git"]),
            ("scene:trading", ["回测", "策略", "胜率", "盈亏比", "夏普",
                              "walk-forward", "参数优化", "信号"]),
            ("scene:project", ["项目", "进度", "里程碑", "需求", "架构"]),
            ("scene:research", ["研究", "分析", "推理", "mcts", "深度", "产业链"]),
        ]
        for tag, keywords in scene_patterns:
            if any(kw in c for kw in keywords):
                return tag
        return None

    def _infer_entity_tags_from_content(self, content: str) -> List[str]:
        """从内容中推断 AMAP 实体标签"""
        tags = []
        c = content
        code_match = re.findall(r'\b(?:00|30|60|68)\d{3}\b', c)
        for code in code_match:
            tags.append(f"entity|object:stock_{code}")
        tool_match = re.findall(r'tools/[\w_]+|scripts/[\w_]+|src/[\w./]+', c)
        for t in tool_match:
            clean = t.replace('/', '_').replace('.py', '')
            tags.append(f"entity|resource:{clean}")
        return list(set(tags))

    def _parse_entity_tags(self, tags: List[str]) -> Dict[str, List[Dict]]:
        """解析 entity| 和 relation| 标签为结构化数据"""
        entities = []
        relations = []
        for tag in tags:
            if tag.startswith("entity|"):
                parts = tag.split(":", 1)
                if len(parts) == 2:
                    entities.append({"name": parts[1], "type": parts[0].replace("entity|", "")})
            elif tag.startswith("relation|"):
                parts = tag.split(":", 1)
                if len(parts) == 2:
                    relations.append({"target": parts[1]})
        return {"entities": entities, "relations": relations}

    def _tag_recall(self, tag: str, limit: int = 3, query: str = "") -> List:
        """按标签召回最近的记忆"""
        try:
            import subprocess
            payload = json.dumps({
                "query": query or tag,
                "tags": [tag],
                "limit": limit,
            })
            r = subprocess.run(
                ["curl", "-s", "-X", "POST",
                 "http://127.0.0.1:9177/v1/default/banks/hermes/memories/recall",
                 "-H", "Content-Type: application/json", "-d", payload],
                capture_output=True, timeout=10,
            )
            if r.returncode == 0 and r.stdout:
                data = json.loads(r.stdout)
                return data if isinstance(data, list) else []
        except Exception:
            pass
        return []

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name == "hindsight_retain":
            content = args.get("content", "")
            if not content:
                return tool_error("Missing required parameter: content")
            context = args.get("context")
            # TiMEM P1: auto-infer scene and inject as tag (not content prefix)
            scene_tag = self._infer_scene(content)
            user_tags = list(args.get("tags") or [])
            if scene_tag and scene_tag not in user_tags:
                user_tags.append(scene_tag)

            # P3.1: Auto-infer AMAP entity tags from content
            if not"--no-auto-entity" in (args.get("tags") or []):
                auto_entity_tags = self._infer_entity_tags_from_content(content)
                for t in auto_entity_tags:
                    if t not in user_tags:
                        user_tags.append(t)

            # Parse entity/relation tags for associative memory
            parsed = self._parse_entity_tags(user_tags)
            if parsed["entities"] or parsed["relations"]:
                logger.debug(
                    "Entity-aware retain: entities=%s, relations=%s",
                    [e["name"] for e in parsed["entities"]],
                    [r["target"] for r in parsed["relations"]],
                )

            # ── 语义去重：retain 前检查是否已有近重复内容 ──
            dedup_enabled = self._config.get("dedup", True)
            if dedup_enabled and content.strip():
                try:
                    import subprocess
                    dedup_script = os.path.join(get_hermes_home(), "hindsight", "memory_dedup.py")
                    r = subprocess.run(
                        ["python3", dedup_script, "filter"],
                        input=content.encode("utf-8"),
                        capture_output=True, timeout=10,
                    )
                    stdout = r.stdout.decode("utf-8", errors="replace").strip()
                    # returncode: 0=唯一  1=严格重复(≥0.85)  2=近似(≥0.70)
                    if r.returncode in (1, 2):
                        label = "duplicate" if r.returncode == 1 else "approximate"
                        score_str = stdout.split("|")[1] if "|" in stdout else "?"
                        logger.debug("hindsight_retain: skipped (%s, score=%s)", label, score_str)
                        return json.dumps({"result": f"Memory already exists ({label}, deduplicated)."})
                except Exception as e:
                    logger.debug("hindsight_retain: dedup check failed, proceeding: %s", e)

            # ── Entity-based dedup: if same entity tags exist within 24h, skip ──
            try:
                from datetime import datetime, timezone, timedelta
                entity_dedup_hours = int(self._config.get("entity_dedup_window_hours", 24))
                if auto_entity_tags and entity_dedup_hours > 0:
                    for entity_tag in auto_entity_tags:
                        recent_entity = self._tag_recall(entity_tag, limit=3, query="")
                        now_utc = datetime.now(timezone.utc)
                        has_recent = False
                        for r in recent_entity:
                            created_str = getattr(r, "mentioned_at", None) or getattr(r, "created_at", None) or ""
                            if created_str:
                                try:
                                    created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                                    if (now_utc - created) < timedelta(hours=entity_dedup_hours):
                                        has_recent = True
                                        break
                                except (ValueError, TypeError):
                                    continue
                        if has_recent:
                            logger.debug("hindsight_retain: entity dedup hit for %s, skipping", entity_tag)
                            return json.dumps({"result": f"Memory already exists (entity dedup: {entity_tag})."})
            except Exception as e:
                logger.debug("hindsight_retain: entity dedup failed, proceeding: %s", e)

            # ── raw 模式：摘要→hindsight，全文→wiki ──
            if args.get("raw", False):
                try:
                    # 内联摘要提取
                    import re as _re
                    raw_text = content.strip()
                    if len(raw_text) > 200:
                        lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
                        key_lines = []
                        for line in lines:
                            if _re.match(r'^[═\-—=#*▶▸→\s]+$', line):
                                continue
                            if any(kw in line for kw in ['结论','结果','确认','决定','方案',
                                                          '修复','改为','设置','配置',
                                                          '策略','规则','原则','偏好',
                                                          '注意','风险','问题','原因']):
                                key_lines.append(line)
                            elif any(kw in line for kw in ['：',':']) and len(line) > 5:
                                key_lines.append(line)
                        if key_lines:
                            summary = '；'.join(key_lines[:3])
                            if len(summary) > 200:
                                summary = summary[:197] + '...'
                        else:
                            for line in lines:
                                if len(line) > 10 and not _re.match(r'^[═\-—=\s]+$', line):
                                    summary = line[:197] + '...' if len(line) > 200 else line
                                    break
                            else:
                                summary = raw_text[:197] + '...'
                    else:
                        summary = raw_text

                    # 写全文到 wiki
                    from datetime import date
                    import os as _os, tempfile
                    today = date.today().isoformat()
                    vault = _os.environ.get('OBSIDIAN_VAULT_PATH', _os.path.expanduser('~/wiki'))
                    dir_path = _os.path.join(vault, 'agent-memory', 'raw', today)
                    _os.makedirs(dir_path, exist_ok=True)

                    import hashlib
                    hkey = "raw_" + hashlib.md5(raw_text.encode()).hexdigest()[:12]
                    file_path = _os.path.join(dir_path, f'{hkey}.md')

                    wiki_content = f"""---
type: memory_raw
date: {today}
hash: {hkey}
tags: [{scene_tag or 'memory'}]
---

{raw_text}
"""
                    fd, tmp = tempfile.mkstemp(dir=dir_path, suffix='.tmp', prefix='.mem_')
                    try:
                        with _os.fdopen(fd, 'w', encoding='utf-8') as f:
                            f.write(wiki_content)
                            f.flush()
                            _os.fsync(f.fileno())
                        _os.replace(tmp, file_path)
                    except BaseException:
                        try:
                            _os.unlink(tmp)
                        except _os.error:
                            pass
                        raise

                    # 替换 content 为摘要+指针
                    content = f"{summary}\n\n📄 obsidian:agent-memory/raw/{today}/{hkey}.md"
                    if "obsidian" not in user_tags:
                        user_tags.append("obsidian")
                    logger.debug("hindsight_retain raw mode: stored full text to %s", file_path)
                except Exception as e:
                    logger.warning("hindsight_retain raw mode failed, falling back to normal: %s", e)

            try:
                retain_kwargs = self._build_retain_kwargs(
                    content,
                    context=context,
                    tags=user_tags,
                )
                logger.debug("Tool hindsight_retain: bank=%s, content_len=%d, context=%s",
                             self._bank_id, len(content), context)
                self._run_hindsight_operation(lambda client: client.aretain(**retain_kwargs))
                logger.debug("Tool hindsight_retain: success")
                return json.dumps({"result": "Memory stored successfully."})
            except Exception as e:
                logger.warning("hindsight_retain failed: %s", e, exc_info=True)
                return tool_error(f"Failed to store memory: {e}")

        elif tool_name == "hindsight_recall":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            try:
                recall_kwargs: dict = {
                    "bank_id": self._bank_id, "query": query, "budget": self._budget,
                    "max_tokens": self._recall_max_tokens,
                }
                # 优先使用调用时传入 tags，fallback 到 config 级别 _recall_tags
                call_tags = args.get("tags")
                if call_tags:
                    recall_kwargs["tags"] = call_tags
                    recall_kwargs["tags_match"] = self._recall_tags_match
                elif self._recall_tags:
                    recall_kwargs["tags"] = self._recall_tags
                    recall_kwargs["tags_match"] = self._recall_tags_match
                if self._recall_types:
                    recall_kwargs["types"] = self._recall_types
                logger.debug("Tool hindsight_recall: bank=%s, query_len=%d, budget=%s",
                             self._bank_id, len(query), self._budget)
                resp = self._run_hindsight_operation(lambda client: client.arecall(**recall_kwargs))
                num_results = len(resp.results) if resp.results else 0
                logger.debug("Tool hindsight_recall: %d results", num_results)
                if not resp.results:
                    return json.dumps({"result": "No relevant memories found."})
                lines = []
                for i, r in enumerate(resp.results, 1):
                    r_tags = getattr(r, 'tags', None) or []
                    scene_tags = [t for t in r_tags if isinstance(t, str) and t.startswith("scene:")]
                    tag_str = f" [{','.join(scene_tags[:3])}]" if scene_tags else ""
                    lines.append(f"{i}. {r.text}{tag_str}")
                return json.dumps({"result": "\n".join(lines)})
            except Exception as e:
                logger.warning("hindsight_recall failed: %s", e, exc_info=True)
                return tool_error(f"Failed to search memory: {e}")

        elif tool_name == "hindsight_reflect":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            try:
                logger.debug("Tool hindsight_reflect: bank=%s, query_len=%d, budget=%s",
                             self._bank_id, len(query), self._budget)
                resp = self._run_hindsight_operation(
                    lambda client: client.areflect(
                        bank_id=self._bank_id, query=query, budget=self._budget
                    )
                )
                logger.debug("Tool hindsight_reflect: response_len=%d", len(resp.text or ""))
                return json.dumps({"result": resp.text or "No relevant memories found."})
            except Exception as e:
                logger.warning("hindsight_reflect failed: %s", e, exc_info=True)
                return tool_error(f"Failed to reflect: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        """Refresh cached per-session state when the agent rotates session_id.

        Fires on /resume, /branch, /reset, /new, and context compression.
        Without this hook, initialize()-cached state (``_session_id``,
        ``_document_id``, ``_session_turns``, ``_turn_counter``) would keep
        pointing at the previous session and writes would land in the wrong
        document. See hermes-agent#6672.

        Always update ``_session_id`` so metadata and tags on subsequent
        retains reflect the active session. Always mint a fresh
        ``_document_id`` so the new session's retain doesn't overwrite the
        old session's document on vectorize-io/hindsight#1303. Always clear
        the accumulated batch buffers (``_session_turns``, ``_turn_counter``,
        ``_turn_index``) — even for /resume and /branch, the new session's
        batching must start from zero so an in-flight retain doesn't flush
        under the wrong ``_document_id``.

        Before clearing, flush any buffered turns under the *old*
        ``_document_id``. Users who set ``retain_every_n_turns > 1`` would
        otherwise silently lose whatever's in ``_session_turns`` at the
        moment of switch — the same data-loss class as the shutdown race,
        just at a different lifecycle event.

        Also wait for any in-flight prefetch from the old session and drop
        its cached result; otherwise the new session's first ``prefetch()``
        could read stale recall text from before the switch.

        ``parent_session_id`` is recorded for lineage tags on future retains.
        ``reset`` is accepted but not needed for Hindsight's state model —
        buffer clearing is correct for every session switch, not only /reset.
        """
        new_id = str(new_session_id or "").strip()
        if not new_id:
            return

        # 1. Flush any buffered turns under the OLD identifiers. Snapshot
        # everything before mutating self._* so metadata + tags + doc_id
        # all reference the old session consistently.
        if self._session_turns:
            old_turns = list(self._session_turns)
            old_session_id = self._session_id
            old_parent_session_id = self._parent_session_id
            old_turn_index = self._turn_index
            old_metadata = self._build_metadata(
                message_count=len(old_turns) * 2,
                turn_index=old_turn_index,
            )
            old_lineage_tags: list[str] = []
            if old_session_id:
                old_lineage_tags.append(f"session:{old_session_id}")
            if old_parent_session_id:
                old_lineage_tags.append(f"parent:{old_parent_session_id}")
            old_content = "[" + ",".join(old_turns) + "]"
            # Resolve doc_id + update_mode against the OLD session BEFORE
            # we rotate _session_id, so the flush lands in the old
            # session's document either way (legacy: per-process unique;
            # ≥0.5.0: stable session-scoped + append).
            old_document_id, old_update_mode = self._resolve_retain_target(
                self._document_id
            )

            def _flush():
                try:
                    item = self._build_retain_kwargs(
                        old_content,
                        context=self._retain_context,
                        metadata=old_metadata,
                        tags=old_lineage_tags or None,
                    )
                    item.pop("bank_id", None)
                    item.pop("retain_async", None)
                    if old_update_mode is not None:
                        item["update_mode"] = old_update_mode
                    logger.debug(
                        "Hindsight flush-on-switch: bank=%s, doc=%s, mode=%s, num_turns=%d",
                        self._bank_id, old_document_id, old_update_mode, len(old_turns),
                    )
                    self._run_hindsight_operation(
                        lambda client: client.aretain_batch(
                            bank_id=self._bank_id,
                            items=[item],
                            document_id=old_document_id,
                            retain_async=self._retain_async,
                        )
                    )
                except Exception as e:
                    logger.warning("Hindsight flush-on-switch failed: %s", e, exc_info=True)

            # Route the flush through the same writer queue sync_turn
            # uses. That serializes it behind any still-queued retains
            # from the old session (FIFO by document_id), avoids racing
            # two threads on aretain_batch against the same document, and
            # keeps shutdown's drain semantics intact. Skip enqueue if
            # shutdown has already fired — the writer is draining/gone.
            if not self._shutting_down.is_set():
                self._ensure_writer()
                self._register_atexit()
                self._retain_queue.put(_flush)

        # 2. Drain any in-flight prefetch from the old session and drop
        # its cached result so the new session doesn't see stale recall.
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            self._prefetch_result = ""

        # 3. Now rotate to the new session.
        if parent_session_id:
            self._parent_session_id = str(parent_session_id).strip()
        self._session_id = new_id
        start_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._document_id = f"{self._session_id}-{start_ts}"
        self._session_turns = []
        self._turn_counter = 0
        self._turn_index = 0
        logger.debug(
            "Hindsight on_session_switch: new_session=%s parent=%s reset=%s doc=%s",
            self._session_id, self._parent_session_id, reset, self._document_id,
        )

    def shutdown(self) -> None:
        logger.debug("Hindsight shutdown: stopping writer + waiting for background threads")
        # Stop accepting new retain jobs first so anyone still calling
        # sync_turn() during teardown is dropped, not enqueued.
        self._shutting_down.set()
        # Drain the writer: it will finish in-flight work, then exit on
        # the sentinel. Bounded join keeps shutdown predictable even if
        # the daemon is wedged.
        writer = self._writer_thread
        if writer is not None and writer.is_alive():
            try:
                self._retain_queue.put(_WRITER_SENTINEL)
            except Exception:
                pass
            writer.join(timeout=10.0)
            if writer.is_alive():
                logger.warning(
                    "Hindsight writer did not stop within 10s; "
                    "abandoning %d pending retain(s)",
                    self._retain_queue.qsize(),
                )
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=5.0)
        if self._client is not None:
            try:
                if self._mode == "local_embedded":
                    # HindsightEmbedded.close() delegates to its sync client.close().
                    # When Hermes created/used that client on the shared async loop,
                    # closing it from this thread can raise "attached to a different
                    # loop" before aiohttp releases the session. Close the embedded
                    # inner async client on the shared loop first, then let the
                    # wrapper clean up daemon/UI bookkeeping.
                    inner_client = getattr(self._client, "_client", None)
                    if inner_client is not None and hasattr(inner_client, "aclose"):
                        _run_sync(inner_client.aclose())
                        try:
                            self._client._client = None
                        except Exception:
                            pass
                    try:
                        self._client.close()
                    except RuntimeError:
                        pass
                else:
                    self._run_sync(self._client.aclose())
            except Exception:
                pass
            self._client = None
        # The module-global background event loop (_loop / _loop_thread)
        # is intentionally NOT stopped here. It is shared across every
        # HindsightMemoryProvider instance in the process — the plugin
        # loader creates a new provider per AIAgent, and the gateway
        # creates one AIAgent per concurrent chat session. Stopping the
        # loop from one provider's shutdown() strands the aiohttp
        # ClientSession + TCPConnector owned by every sibling provider
        # on a dead loop, which surfaces as the "Unclosed client session"
        # / "Unclosed connector" warnings reported in #11923. The loop
        # runs on a daemon thread and is reclaimed on process exit;
        # per-session cleanup happens via self._client.aclose() above.


def register(ctx) -> None:
    """Register Hindsight as a memory provider plugin."""
    ctx.register_memory_provider(HindsightMemoryProvider())
