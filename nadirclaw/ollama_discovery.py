"""Ollama auto-discovery for NadirClaw.

Automatically discovers Ollama instances on the local network by scanning
common ports and hostnames.
"""

import concurrent.futures
import json
import socket
import urllib.request
from typing import List, Optional, Tuple

DEFAULT_OLLAMA_PORT = 11434
DISCOVERY_TIMEOUT = 2  # seconds per host


def _check_ollama_at(host: str, port: int = DEFAULT_OLLAMA_PORT) -> Optional[dict]:
    """Check if Ollama is running at a specific host:port.

    Returns dict with endpoint info if successful, None otherwise.
    """
    url = f"http://{host}:{port}/api/tags"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=DISCOVERY_TIMEOUT) as resp:
            data = json.loads(resp.read())
            # Validate it's actually Ollama by checking response structure
            if "models" in data:
                model_count = len(data.get("models", []))
                return {
                    "host": host,
                    "port": port,
                    "url": f"http://{host}:{port}",
                    "model_count": model_count,
                }
    except Exception:
        pass
    return None


def _get_local_ip_prefix() -> Optional[str]:
    """Get the local network prefix (e.g., '192.168.1') for scanning."""
    try:
        # Create a socket to get local IP without actually connecting
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        # Use a dummy external address (doesn't actually connect)
        s.connect(("10.255.255.255", 1))
        local_ip = s.getsockname()[0]
        s.close()
        # Extract network prefix (first 3 octets)
        parts = local_ip.split(".")
        if len(parts) == 4:
            return ".".join(parts[:3])
    except Exception:
        pass
    return None


def discover_ollama_instances(scan_network: bool = False) -> List[dict]:
    """Discover Ollama instances on localhost and optionally the local network.

    Args:
        scan_network: If True, scans common hosts on the local subnet (slower).

    Returns:
        List of dicts with keys: host, port, url, model_count.
        Sorted by model_count (descending).
    """
    candidates = [
        "localhost",
        "127.0.0.1",
        socket.gethostname(),  # This machine's hostname
    ]

    # Add common Docker/VM hosts
    candidates.extend([
        "host.docker.internal",
        "192.168.65.2",  # Docker Desktop on macOS
    ])

    if scan_network:
        # Scan local subnet (e.g., 192.168.1.1-254)
        prefix = _get_local_ip_prefix()
        if prefix:
            # Scan a smaller range for speed (common router/server IPs)
            scan_range = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 254]
            candidates.extend([f"{prefix}.{i}" for i in scan_range])

    # Deduplicate
    unique_candidates = []
    seen = set()
    for host in candidates:
        if host not in seen:
            seen.add(host)
            unique_candidates.append(host)

    # Parallel scan with ThreadPoolExecutor
    found = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(_check_ollama_at, host): host
            for host in unique_candidates
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                found.append(result)

    # Sort by model count (prefer instances with more models)
    found.sort(key=lambda x: x["model_count"], reverse=True)
    return found


def discover_best_ollama() -> Optional[dict]:
    """Quick discovery: check localhost first, fallback to network scan.

    Returns the best Ollama instance (most models), or None if not found.
    """
    # Fast path: check localhost first
    local_result = _check_ollama_at("localhost")
    if local_result:
        return local_result

    # Fallback: scan network (slower)
    instances = discover_ollama_instances(scan_network=True)
    return instances[0] if instances else None


def format_discovery_results(instances: List[dict]) -> str:
    """Format discovery results as a human-readable string."""
    if not instances:
        return "No Ollama instances found."

    lines = [f"Found {len(instances)} Ollama instance(s):\n"]
    for i, inst in enumerate(instances, 1):
        models = "model" if inst["model_count"] == 1 else "models"
        lines.append(
            f"  {i}. {inst['url']:30s} ({inst['model_count']} {models})"
        )
    return "\n".join(lines)
