# NadirClaw GitHub Action

Use NadirClaw in your CI/CD pipelines to automatically route LLM calls to the cheapest model that can handle each request.

## Repository

[`doramirdor/nadirclaw-action`](https://github.com/doramirdor/nadirclaw-action)

## Quick Start

```yaml
steps:
  - uses: doramirdor/nadirclaw-action@v1
    with:
      anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
      google-api-key: ${{ secrets.GOOGLE_API_KEY }}

  # All subsequent steps automatically route through NadirClaw
  # via OPENAI_BASE_URL=http://localhost:8856/v1
  - name: AI-powered step
    run: python my_ai_script.py
```

## What It Does

1. Installs NadirClaw in the GitHub Actions runner
2. Starts the routing proxy on localhost
3. Sets `OPENAI_BASE_URL` for all subsequent steps
4. Every LLM call gets classified and routed to the optimal model

Simple prompts go to cheap models. Complex prompts go to premium. You save 50-70% on typical CI workloads without sacrificing quality where it matters.

## Use Cases

- **AI code review** on pull requests
- **Automated test generation**
- **Documentation generation**
- **Code migration assistants**
- **Security scanning with LLMs**

Any CI step that makes LLM API calls benefits from routing.

## Full Documentation

See the [nadirclaw-action README](https://github.com/doramirdor/nadirclaw-action) for all inputs, outputs, and example workflows.
