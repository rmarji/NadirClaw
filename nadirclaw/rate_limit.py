"""Per-model rate limiting for NadirClaw.

Provides a sliding-window rate limiter keyed by model name.
Configured via environment variables:

  NADIRCLAW_MODEL_RATE_LIMITS  — comma-separated model=rpm pairs
      e.g. "gemini-3-flash-preview=30,gpt-4.1=60"

  NADIRCLAW_DEFAULT_MODEL_RPM  — default max requests/minute for
      any model not listed above. 0 or unset means no default limit.

Rate-limited requests raise RateLimitExhausted so the fallback chain
can try the next model.
"""

import collections
import os
import logging
import time
from threading import Lock
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("nadirclaw")


class ModelRateLimiter:
    """Sliding-window rate limiter keyed by model name.

    Thread-safe. Each model has its own deque of timestamps and a
    configured max-requests-per-minute limit.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        # model -> deque of timestamps
        self._hits: Dict[str, collections.deque] = {}
        # model -> max rpm (0 = unlimited)
        self._limits: Dict[str, int] = {}
        self._default_rpm: int = 0
        self._reload_config()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _reload_config(self) -> None:
        """Parse config from environment variables."""
        raw = os.getenv("NADIRCLAW_MODEL_RATE_LIMITS", "")
        limits: Dict[str, int] = {}
        for pair in raw.split(","):
            pair = pair.strip()
            if "=" not in pair:
                continue
            model, rpm_str = pair.rsplit("=", 1)
            model = model.strip()
            try:
                rpm = int(rpm_str.strip())
                if rpm > 0:
                    limits[model] = rpm
            except ValueError:
                logger.warning("Invalid rate limit config: %s", pair)
        self._limits = limits

        default_str = os.getenv("NADIRCLAW_DEFAULT_MODEL_RPM", "0")
        try:
            self._default_rpm = max(0, int(default_str))
        except ValueError:
            self._default_rpm = 0

    def reload(self) -> None:
        """Reload configuration from environment. Clears all counters."""
        with self._lock:
            self._hits.clear()
            self._reload_config()

    def set_limit(self, model: str, rpm: int) -> None:
        """Programmatically set a per-model limit (for testing)."""
        with self._lock:
            if rpm > 0:
                self._limits[model] = rpm
            else:
                self._limits.pop(model, None)

    def set_default(self, rpm: int) -> None:
        """Programmatically set the default limit (for testing)."""
        with self._lock:
            self._default_rpm = max(0, rpm)

    def get_limit(self, model: str) -> int:
        """Return the effective RPM limit for a model. 0 = unlimited."""
        return self._limits.get(model, self._default_rpm)

    # ------------------------------------------------------------------
    # Rate check
    # ------------------------------------------------------------------

    def check(self, model: str) -> Optional[int]:
        """Check if a model request is allowed.

        Returns None if allowed (and records the hit).
        Returns seconds-until-retry if rate-limited.
        """
        limit = self.get_limit(model)
        if limit <= 0:
            return None  # No limit configured

        now = time.time()
        window = 60  # 1 minute sliding window

        with self._lock:
            q = self._hits.setdefault(model, collections.deque())

            # Evict timestamps outside the window
            while q and q[0] <= now - window:
                q.popleft()

            if len(q) >= limit:
                retry_after = int(q[0] + window - now) + 1
                return max(1, retry_after)

            q.append(now)
            return None

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return current rate limit status for all configured models."""
        now = time.time()
        window = 60
        models_status = {}

        with self._lock:
            # Snapshot under lock so limits and hits are consistent
            all_models = set(self._limits.keys()) | set(self._hits.keys())
            for model in sorted(all_models):
                limit = self._limits.get(model, self._default_rpm)
                q = self._hits.get(model, collections.deque())
                recent = sum(1 for t in q if t > now - window)
                models_status[model] = {
                    "rpm_limit": limit if limit > 0 else "unlimited",
                    "current_rpm": recent,
                    "remaining": max(0, limit - recent) if limit > 0 else "unlimited",
                }
            default_rpm = self._default_rpm

        return {
            "default_rpm": default_rpm if default_rpm > 0 else "unlimited",
            "models": models_status,
        }

    def reset(self, model: Optional[str] = None) -> None:
        """Clear hit counters. If model is given, clear only that model."""
        with self._lock:
            if model:
                self._hits.pop(model, None)
            else:
                self._hits.clear()


# Singleton
_model_rate_limiter: Optional[ModelRateLimiter] = None
_init_lock = Lock()


def get_model_rate_limiter() -> ModelRateLimiter:
    """Get the global ModelRateLimiter singleton."""
    global _model_rate_limiter
    if _model_rate_limiter is None:
        with _init_lock:
            if _model_rate_limiter is None:
                _model_rate_limiter = ModelRateLimiter()
    return _model_rate_limiter
