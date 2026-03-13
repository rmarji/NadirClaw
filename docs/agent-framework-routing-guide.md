# How We Cut Our AI Costs 60% With a 200-Line Routing Patch

*A practical guide to intelligent model routing for AI agent frameworks*

---

## The Problem

We run a fleet of AI agents. Each agent handles everything from "what's 2+2?" to "design a distributed authentication system with CRDT conflict resolution." Every single request was hitting Claude Opus — our most expensive model — because the agent framework sends massive payloads (system prompts, tool schemas, conversation history) that made every request *look* complex to naive routers.

The math was brutal: ~$0.93 per complex request × hundreds of requests per day = pain.

## What We Tried First

**LiteLLM's built-in complexity_router.** It crashed. The router tried to classify the entire 104K-token payload (system prompt + 25 tool definitions + the user's actual question), panicked, and threw a `BadRequestError`. The user's "What is 2+2?" was buried under kilobytes of JSON schema.

**n8n as a routing proxy.** Workable, but n8n's webhook nodes can't stream SSE responses. For short answers it's fine. For long responses, the user stares at nothing for 30+ seconds. Unacceptable.

## The Solution: NadirClaw + LiteLLM

[NadirClaw](https://github.com/doramirdor/NadirClaw) is an open-source intelligent model router that sits between your agent framework and your LLM provider. It classifies prompt complexity in ~10ms using sentence embeddings, then routes to the appropriate model tier.

The architecture:

```
User Message
  → Agent Framework (sends full payload)
    → NadirClaw (port 8856)
      → Extracts ONLY the last user message
      → Classifies complexity (10-50ms)
      → Picks model tier
        → LiteLLM (port 4000)
          → Actual provider (Anthropic, Google, OpenAI)
            → Response streams back (full SSE)
```

### Why This Works

NadirClaw does what LiteLLM's complexity_router should have done: it extracts the last user message (`role: "user"`) and classifies *only that*. The 100K tokens of system prompt and tool schemas? Ignored.

```python
# From NadirClaw's server.py — the key insight
user_msgs = [m.text_content() for m in messages if m.role == "user"]
prompt = user_msgs[-1] if user_msgs else ""
```

That's it. One line that makes the difference between "crashes on every request" and "routes correctly in 10ms."

## Setup Guide

### Prerequisites

- A running LiteLLM instance (or any OpenAI-compatible API)
- Docker
- Your agent framework configured to use an OpenAI-compatible provider

### Step 1: Define Your Model Tiers

Decide which models handle which complexity levels:

| Tier | Use Case | Example Model | Cost |
|------|----------|---------------|------|
| simple | Greetings, math, lookups | Gemini 2.5 Flash / GPT-4o-mini | $$ |
| mid | General conversation, summaries | Claude Sonnet / GPT-4o | $$$ |
| complex | Architecture, analysis, code | Claude Opus / o1 | $$$$ |
| reasoning | Multi-step logic, proofs | o1-pro / Claude Opus | $$$$$ |

### Step 2: Deploy NadirClaw

```bash
mkdir -p /data/nadirclaw && cd /data/nadirclaw

git clone https://github.com/doramirdor/NadirClaw.git

cat > .env << 'EOF'
# Point NadirClaw at your LiteLLM (or any OpenAI-compatible API)
NADIRCLAW_API_BASE=http://localhost:4000/v1
NADIRCLAW_API_KEY=your-litellm-key

# Map tiers to your model names
# Use openai/ prefix so NadirClaw routes through the API_BASE
NADIRCLAW_SIMPLE_MODEL=openai/your-cheap-model
NADIRCLAW_MID_MODEL=openai/your-mid-model
NADIRCLAW_COMPLEX_MODEL=openai/your-expensive-model
NADIRCLAW_REASONING_MODEL=openai/your-expensive-model
NADIRCLAW_PORT=8856

# These ensure the internal LiteLLM client routes through your proxy
OPENAI_API_KEY=your-litellm-key
OPENAI_API_BASE=http://localhost:4000/v1
EOF

cat > docker-compose.yml << 'EOF'
services:
  nadirclaw:
    build:
      context: ./NadirClaw
      dockerfile: Dockerfile
    container_name: nadirclaw
    restart: unless-stopped
    network_mode: host
    env_file: .env
    command: ["nadirclaw", "serve", "--port", "8856"]
    volumes:
      - ./data:/root/.nadirclaw
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8856/health"]
      interval: 30s
      timeout: 5s
      start_period: 60s
      retries: 3
EOF

docker compose up -d
```

Wait ~30 seconds for the sentence-transformer model to load, then verify:

```bash
curl http://localhost:8856/health
# {"status":"ok","version":"0.10.0","simple_model":"openai/...","complex_model":"openai/..."}
```

### Step 3: Test Classification

NadirClaw has a `/v1/classify` endpoint for dry-run testing (no LLM call):

```bash
# Simple prompt → should route to cheap model
curl -s http://localhost:8856/v1/classify \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "What is 2+2?"}' | python3 -m json.tool

# Complex prompt → should route to expensive model
curl -s http://localhost:8856/v1/classify \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Design a distributed authentication system with token refresh and race condition handling"}' | python3 -m json.tool
```

You should see `"tier": "simple"` for the first and `"tier": "complex"` for the second.

### Step 4: Test End-to-End

```bash
# Non-streaming test
curl -s http://localhost:8856/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is 2+2?"}]}'

# Streaming test
curl -s http://localhost:8856/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Design a distributed auth system"}],"stream":true}'
```

Check the response's `nadirclaw_metadata.routing.tier` to see which tier was selected.

### Step 5: Point Your Agent Framework

NadirClaw exposes a standard OpenAI-compatible API. Point your agent framework at it as if it were any other provider:

```
Base URL: http://your-nadirclaw-host:8856/v1
API Key:  anything (NadirClaw handles auth to the backend)
Model:    auto
```

Your framework sends requests as normal. NadirClaw classifies, picks the right model, and streams the response back. The framework never knows the difference.

## The Session Cache Problem (and How We Fixed It)

After deploying, we noticed something: every message was routing to the "mid" tier, regardless of complexity. Even "Design a distributed system with CRDTs and vector clocks" was going to Sonnet instead of Opus.

### Root Cause

NadirClaw has a **session cache**. It hashes the system prompt + first user message to create a session key. Once classified, that session is pinned to a tier for 30 minutes.

The problem: the first message in every agent session is usually something simple (a greeting, a bootstrap message). That classifies as "mid." Then the cache pins every subsequent message — even genuinely complex ones — to "mid" for the next 30 minutes.

### The Fix: Upgrade-Only Cache

We patched the session cache with three changes:

1. **Always classify** — every message runs through the classifier, even on cache hit
2. **Upgrade only** — if the new classification is a higher tier, upgrade the cache; if lower, keep the cached (higher) tier
3. **Shorter TTL** — 5 minutes instead of 30

```python
# Tier hierarchy
TIER_ORDER = {"simple": 0, "mid": 1, "complex": 2, "reasoning": 3}

def upgrade_if_higher(self, messages, new_model, new_tier):
    """Upgrade cached tier if new tier outranks it. Never downgrade."""
    key = self._make_key(messages)
    new_rank = self.TIER_ORDER.get(new_tier, 0)
    
    entry = self._cache.get(key)
    if entry is None:
        self._cache[key] = (new_model, new_tier, time.time())
        return new_model, new_tier
    
    cached_model, cached_tier, _ = entry
    cached_rank = self.TIER_ORDER.get(cached_tier, 0)
    
    if new_rank > cached_rank:
        # Escalate
        self._cache[key] = (new_model, new_tier, time.time())
        return new_model, new_tier
    
    # Keep existing (equal or higher) tier
    return cached_model, cached_tier
```

The result:

```
Message    Classified  Cached    Actually Used
──────────────────────────────────────────────
"Hi"       simple      (none)    cheap-model      ← new session
"Design    complex     →upgrade  expensive-model  ← escalated
 system"
"Thanks"   simple      complex   expensive-model  ← no downgrade
```

This gives you stability (no jarring model switches) without the downside of missing complex prompts.

**We submitted this as [PR #27](https://github.com/doramirdor/NadirClaw/pull/27) to the upstream NadirClaw repo.**

## Results

After one day of running:

- **Simple prompts** (greetings, math, lookups) → routed to Gemini 2.5 Pro. Cost: ~$0.001/request
- **Complex prompts** (architecture, analysis) → routed to Claude Opus. Cost: ~$0.93/request
- **Classification overhead:** 10-50ms per request (negligible vs. model latency)
- **Streaming:** works perfectly — NadirClaw passes SSE through transparently

The key insight: in an agent framework, most requests are actually simple (tool results, acknowledgments, status checks). Only 10-20% genuinely need the expensive model. The router catches that automatically.

## Gotchas

**1. The `openai/` prefix matters.** NadirClaw uses LiteLLM internally. If your model names are custom aliases (like `claw-reason` instead of `claude-opus-4`), prefix them with `openai/` so NadirClaw's internal LiteLLM client routes through your API base URL instead of trying to call the provider directly.

**2. Environment variable overrides.** If your agent framework sets the model via environment variable (like `OPENCLAW_PRIMARY_MODEL`), that overrides config file settings. Change it at the orchestration layer (Coolify, Docker env, k8s configmap), not in the config file.

**3. Non-streaming timeouts.** Complex prompts routed to Opus can generate massive responses. If you're testing without streaming, set a generous timeout (120s+). In production with streaming, this isn't an issue — first tokens arrive in 1-2 seconds.

**4. Session cache key collisions.** The cache keys on `system_prompt[:200] + first_user_message[:200]`. If multiple users share the same system prompt and happen to start with the same first message, they'll share a cache entry. In practice this is rare, but worth knowing.

**5. Container rebuilds reset patches.** If you patch NadirClaw in the running container (`docker exec ... sed -i`), the patch is lost on container rebuild. For persistence, fork the repo, commit your changes, and point your Dockerfile at your fork.

## Quick Reference

```bash
# Health check
curl http://localhost:8856/health

# Dry-run classification (no LLM call)
curl -s http://localhost:8856/v1/classify \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "your test prompt"}'

# Full request (streaming)
curl -s http://localhost:8856/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"hello"}],"stream":true}'

# Check logs for routing decisions
docker logs nadirclaw --tail 20 | grep -v health
```

## Links

- **NadirClaw:** [github.com/doramirdor/NadirClaw](https://github.com/doramirdor/NadirClaw)
- **LiteLLM:** [github.com/BerriAI/litellm](https://github.com/BerriAI/litellm)
- **Upgrade-only cache PR:** [NadirClaw #27](https://github.com/doramirdor/NadirClaw/pull/27)

---

*This setup runs in production across our agent fleet. The routing is transparent to users — they just notice that simple questions come back faster and cheaper, while complex questions still get the full power of the best models available.*
