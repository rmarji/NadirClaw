"""Tests for fallback chain configuration and behavior."""

import os
import pytest


class TestFallbackChainConfig:
    def test_default_chain_includes_tier_models(self):
        """Default chain should include complex and simple models."""
        from nadirclaw.settings import settings
        chain = settings.FALLBACK_CHAIN
        assert settings.COMPLEX_MODEL in chain
        assert settings.SIMPLE_MODEL in chain
        # Complex should come first
        assert chain.index(settings.COMPLEX_MODEL) < chain.index(settings.SIMPLE_MODEL)

    def test_custom_chain_from_env(self, monkeypatch):
        """NADIRCLAW_FALLBACK_CHAIN env var should override defaults."""
        monkeypatch.setenv("NADIRCLAW_FALLBACK_CHAIN", "model-a,model-b,model-c")
        from nadirclaw.settings import Settings
        s = Settings()
        assert s.FALLBACK_CHAIN == ["model-a", "model-b", "model-c"]

    def test_empty_chain_env_uses_defaults(self, monkeypatch):
        """Empty NADIRCLAW_FALLBACK_CHAIN should fall back to defaults."""
        monkeypatch.setenv("NADIRCLAW_FALLBACK_CHAIN", "")
        from nadirclaw.settings import Settings
        s = Settings()
        assert len(s.FALLBACK_CHAIN) >= 1

    def test_chain_deduplicates(self, monkeypatch):
        """Default chain should not have duplicate models."""
        # When simple == complex, chain should still work
        monkeypatch.setenv("NADIRCLAW_SIMPLE_MODEL", "same-model")
        monkeypatch.setenv("NADIRCLAW_COMPLEX_MODEL", "same-model")
        monkeypatch.delenv("NADIRCLAW_FALLBACK_CHAIN", raising=False)
        from nadirclaw.settings import Settings
        s = Settings()
        assert s.FALLBACK_CHAIN.count("same-model") == 1
