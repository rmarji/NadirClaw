"""Tests for Ollama auto-discovery."""

import json
import socket
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from nadirclaw.ollama_discovery import (
    _check_ollama_at,
    _get_local_ip_prefix,
    discover_best_ollama,
    discover_ollama_instances,
    format_discovery_results,
)


class TestCheckOllamaAt:
    """Tests for _check_ollama_at."""

    def test_success(self):
        """Test successful Ollama detection."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "models": [
                {"name": "llama3.1:8b"},
                {"name": "qwen3:32b"},
            ]
        }).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = _check_ollama_at("localhost", 11434)

        assert result is not None
        assert result["host"] == "localhost"
        assert result["port"] == 11434
        assert result["url"] == "http://localhost:11434"
        assert result["model_count"] == 2

    def test_connection_error(self):
        """Test connection failure."""
        with patch("urllib.request.urlopen", side_effect=ConnectionError):
            result = _check_ollama_at("nonexistent-host", 11434)

        assert result is None

    def test_invalid_response(self):
        """Test invalid JSON response."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"not json"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = _check_ollama_at("localhost", 11434)

        assert result is None

    def test_missing_models_key(self):
        """Test response without 'models' key (not Ollama)."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"error": "not found"}).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = _check_ollama_at("localhost", 11434)

        assert result is None


class TestGetLocalIpPrefix:
    """Tests for _get_local_ip_prefix."""

    def test_success(self):
        """Test successful IP prefix extraction."""
        with patch("socket.socket") as mock_socket:
            mock_instance = MagicMock()
            mock_instance.getsockname.return_value = ("192.168.1.100", 0)
            mock_socket.return_value = mock_instance

            result = _get_local_ip_prefix()

        assert result == "192.168.1"

    def test_socket_error(self):
        """Test socket error handling."""
        with patch("socket.socket", side_effect=OSError):
            result = _get_local_ip_prefix()

        assert result is None


class TestDiscoverOllamaInstances:
    """Tests for discover_ollama_instances."""

    def test_localhost_only(self):
        """Test discovery without network scan."""
        def mock_check(host, port=11434):
            if host in ("localhost", "127.0.0.1"):
                return {
                    "host": host,
                    "port": port,
                    "url": f"http://{host}:{port}",
                    "model_count": 3,
                }
            return None

        with patch("nadirclaw.ollama_discovery._check_ollama_at", side_effect=mock_check):
            results = discover_ollama_instances(scan_network=False)

        # Should find localhost and/or 127.0.0.1
        assert len(results) >= 1
        assert all(r["host"] in ("localhost", "127.0.0.1") for r in results)

    def test_network_scan(self):
        """Test discovery with network scan."""
        def mock_check(host, port=11434):
            if host == "192.168.1.10":
                return {
                    "host": host,
                    "port": port,
                    "url": f"http://{host}:{port}",
                    "model_count": 5,
                }
            elif host == "localhost":
                return {
                    "host": host,
                    "port": port,
                    "url": f"http://{host}:{port}",
                    "model_count": 2,
                }
            return None

        with patch("nadirclaw.ollama_discovery._check_ollama_at", side_effect=mock_check), \
             patch("nadirclaw.ollama_discovery._get_local_ip_prefix", return_value="192.168.1"):
            results = discover_ollama_instances(scan_network=True)

        # Should find both, sorted by model count (192.168.1.10 first)
        assert len(results) >= 2
        assert results[0]["host"] == "192.168.1.10"
        assert results[0]["model_count"] == 5

    def test_no_instances_found(self):
        """Test when no Ollama instances are found."""
        with patch("nadirclaw.ollama_discovery._check_ollama_at", return_value=None):
            results = discover_ollama_instances(scan_network=False)

        assert results == []


class TestDiscoverBestOllama:
    """Tests for discover_best_ollama."""

    def test_localhost_first(self):
        """Test that localhost is checked first (fast path)."""
        mock_localhost = {
            "host": "localhost",
            "port": 11434,
            "url": "http://localhost:11434",
            "model_count": 2,
        }

        with patch("nadirclaw.ollama_discovery._check_ollama_at", return_value=mock_localhost) as mock_check:
            result = discover_best_ollama()

        # Should only call _check_ollama_at once (for localhost)
        assert mock_check.call_count == 1
        assert result == mock_localhost

    def test_network_fallback(self):
        """Test network scan fallback when localhost fails."""
        def mock_check(host, port=11434):
            if host == "localhost":
                return None
            return None  # Will trigger network scan in discover_ollama_instances

        mock_network_result = {
            "host": "192.168.1.5",
            "port": 11434,
            "url": "http://192.168.1.5:11434",
            "model_count": 4,
        }

        with patch("nadirclaw.ollama_discovery._check_ollama_at", side_effect=mock_check), \
             patch("nadirclaw.ollama_discovery.discover_ollama_instances", return_value=[mock_network_result]):
            result = discover_best_ollama()

        assert result == mock_network_result

    def test_none_found(self):
        """Test when no instances are found anywhere."""
        with patch("nadirclaw.ollama_discovery._check_ollama_at", return_value=None), \
             patch("nadirclaw.ollama_discovery.discover_ollama_instances", return_value=[]):
            result = discover_best_ollama()

        assert result is None


class TestFormatDiscoveryResults:
    """Tests for format_discovery_results."""

    def test_empty_results(self):
        """Test formatting when no instances found."""
        output = format_discovery_results([])
        assert output == "No Ollama instances found."

    def test_single_result(self):
        """Test formatting a single instance."""
        instances = [{
            "url": "http://localhost:11434",
            "model_count": 1,
        }]
        output = format_discovery_results(instances)
        assert "Found 1 Ollama instance" in output
        assert "http://localhost:11434" in output
        assert "1 model" in output

    def test_multiple_results(self):
        """Test formatting multiple instances."""
        instances = [
            {"url": "http://192.168.1.10:11434", "model_count": 5},
            {"url": "http://localhost:11434", "model_count": 2},
        ]
        output = format_discovery_results(instances)
        assert "Found 2 Ollama instance(s)" in output
        assert "http://192.168.1.10:11434" in output
        assert "5 models" in output
        assert "http://localhost:11434" in output
        assert "2 models" in output
