# Week 1 v2: RAG API + Crypto Triage Multi-Agent System

This folder started as the class's minimal `/ask` demo and grew into a small portfolio
piece: a grounded RAG API, plus a Google ADK multi-agent system built on top of it for
the Session 3 capstone ("classify a crypto question and route it to the right
specialist"). Everything here is live and real — no mocked tools, no stubbed APIs.

## What's in here

| Piece | What it does |
|---|---|
| **`/ask` RAG API** | Answers questions strictly from an ingested knowledge base (Pinecone), with citations, confidence, cost/latency, and a refusal when the docs don't cover the question. Includes a validate-and-retry demo against malformed structured output. |
| **`/agent` multi-agent system** | A router agent (`agent_app.py`) classifies a message and calls one to three specialists in the same turn: conceptual/technical questions → RAG, live prices → CoinGecko, account issues → a human-approved ticket-filing flow. |
| **Streamlit UI** | Three pages: the RAG demo (ingest + ask), a live Think → Act → Observe trace of the agent (with an Approve/Reject button for ticket filing), and a golden-set eval runner. |

### RAG API (`main.py`)

A typed FastAPI service. `/ask` embeds the question, retrieves the top-k chunks from
Pinecone, and asks the model to answer **using only that context** — if the context
doesn't cover it, it must refuse rather than guess. The response includes:

- `answer` — structured `{answer, confidence, sources_needed, citations}`
- `tokens_used`, `model`, `latency_ms`, `cost_usd` — real runtime metadata, not estimates
- `attempts` — validation/retry log (see the `force_bad` demo below)
- `sources` — the actual retrieved chunks, so you can see what was (and wasn't) used

`/ingest` chunks (via `RecursiveCharacterTextSplitter`), embeds
(`text-embedding-3-small`), and upserts into Pinecone, replacing any previous chunks for
that `document_id`. `/debug/retrieve` runs retrieval only, no generation, useful for
checking the right chunks come back before trusting `/ask`'s answer.

### Crypto Triage Multi-Agent System (`agent_app.py`)

Built for the Session 3 capstone: *"classify a crypto question as conceptual/technical,
market-data, or account-specific, and route it to the matching specialist."*

```
router_agent (crypto_triage_router)
├── crypto_knowledge_agent  → search_docs        (real Pinecone RAG, same index as /ask)
├── market_agent            → get_crypto_price   (real CoinGecko public API, no key needed)
└── escalation_agent        → draft_ticket (local, safe)
                             → execute_sql (real Supabase hosted MCP server — gated, see below)
```

Three specialists, three genuinely different integration styles (direct function call,
plain HTTP API, MCP protocol) — a router earns its keep here rather than being three
agents pretending to need one.

**Compound questions get a real combined answer.** `crypto_knowledge_agent` and
`market_agent` run in `mode="single_turn"`, so the router calls them as tools and keeps
control instead of permanently handing off the turn. This matters: with a plain
`transfer_to_agent` handoff, whichever specialist got control would silently answer the
*other* half of a compound question ("ETH price, and when was the Merge?") from its own
training data instead of refusing or delegating — no tool call, and often still
factually correct on famous facts, which is exactly what made it easy to miss in testing.
`escalation_agent` deliberately keeps full-transfer mode, since its draft → approve flow
needs a real multi-turn session, not a single-turn call.

**Human-in-the-loop on the one consequential action.** Filing a ticket
(`execute_sql`, via Supabase's hosted MCP server, `https://mcp.supabase.com/mcp`) is
blocked by a `before_tool_callback` unless session state carries
`escalation_approved=True` — set *only* by a human clicking **Approve & File** in the
Streamlit trace page, never by the model. The callback also rejects anything that isn't
a single `INSERT INTO tickets (...)` statement, since `execute_sql` can otherwise run
arbitrary SQL. This is a structural gate, not a prompted instruction, and it's been
adversarially tested — see [`prompt_injection_test.md`](prompt_injection_test.md): one
attack (a spoofed `[SYSTEM]` approval claim) was caught by the model's own judgment; a
sharper one (replaying the exact phrase the real Approve button sends) *fooled the model
into believing approval had happened*, and the write was still blocked because approval
isn't something the model can grant itself. Both outcomes were verified with an
independent `SELECT` against the live table, not just the model's own narration.

The ticket tool went through the hosted Supabase MCP server (not a local
filesystem/npx-spawned MCP server) specifically so it works identically on Render (no
Node, ephemeral disk) and locally.

## File map

```text
week-1v2/
├── README.md
├── main.py                         # FastAPI: /ask, /ingest, /debug/retrieve, /agent, /health*
├── agent_app.py                    # ADK router + 3 specialists (crypto triage)
├── demo_page.py                    # Streamlit: RAG ingest + ask demo (entry point)
├── pages/
│   ├── 1_Golden_Set_Eval.py        # Runs golden_set.py against the live API, logs results/
│   └── 2_Agent_Trace.py            # Live Think/Act/Observe trace + ticket Approve/Reject
├── golden_set.py                   # Fixed question set + grading (retrieval/faithfulness/correctness)
├── ingest_corpus.py                # Batch-ingest every data/*.txt via /ingest
├── prompt_injection_test.py        # Adversarial test script for the escalation HITL gate
├── prompt_injection_test.md        # Write-up of the two attacks and results
├── agent_loop_proof.md             # Notes on the ADK event-stream → Think/Act/Observe mapping
├── smoke_test.py                   # No-token API startup check (health + /docs only)
├── data/                           # Crypto knowledge base source docs (ingested into Pinecone)
├── results/                        # Golden-set eval run logs + running summary.md
├── shots/                          # Screenshots of the golden path (UI, trace, escalation flow)
├── requirements.txt
├── .env.example
└── stages/                         # Optional teaching references: how /ask was built up in 3 steps
    ├── stage_1_bare_ask.py
    ├── stage_2_structured_output.py
    └── stage_3_guardrails_and_observability.py
```

## Environment variables

Copy `.env.example` to `.env` and fill these in:

| Variable | Used by | Required for |
|---|---|---|
| `OPENAI_API_KEY` | `main.py` | Embeddings + `/ask` generation (always required) |
| `PINECONE_API_KEY` | `main.py` | Vector store for `/ask`, `/ingest`, `/debug/retrieve`, and `search_docs` |
| `PINECONE_INDEX_NAME` | `main.py` | Same — defaults to `week1v2-rag` if unset |
| `GOOGLE_API_KEY` | `agent_app.py` | `/agent` and the Streamlit Agent Trace page (Gemini) |
| `SUPABASE_ACCESS_TOKEN` | `agent_app.py` | Filing tickets via the hosted Supabase MCP server |
| `SUPABASE_PROJECT_REF` | `agent_app.py` | Same |

Missing `GOOGLE_API_KEY`/Supabase vars only breaks the agent side — `/ask` and the RAG
demo page work fine without them. Missing Supabase vars specifically only breaks
*filing* a ticket; drafting still works and the app prints a warning at import time
instead of failing.

## Running it locally (no Render)

From this `week-1v2` folder:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
test -f .env || cp .env.example .env
# then edit .env and fill in the keys above
```

**Terminal 1 — the API** (serves both `/ask` and `/agent`):

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Confirm it's up without spending tokens: `curl http://127.0.0.1:8000/health`. Full
interactive docs at `http://127.0.0.1:8000/docs`.

**One-time — ingest the knowledge base** (needed before `/ask`, `/debug/retrieve`, or
`search_docs` will find anything):

```bash
python ingest_corpus.py
```

**Terminal 2 — the Streamlit app:**

```bash
streamlit run demo_page.py
```

Open `http://localhost:8501`. The sidebar's **API base URL** field points every page at
the API — leave it as `http://127.0.0.1:8000` for local use. Use the sidebar page picker
to switch between:

- **main page** — ingest text, ask questions, toggle the guardrail retry demo
- **Golden Set Eval** — run the fixed question set against the live API and log
  retrieval-hit/faithfulness/correctness rates to `results/` for comparing runs over time
- **Agent Trace** — talk to the crypto triage router, watch each tool call live, and
  approve or reject any escalation ticket it drafts

### Try the guardrail demo

Check **Force a bad first response to demo validation + retry** on the main page. The
API deliberately asks the model for malformed JSON first, fails Pydantic validation,
logs the failure in `attempts`, and retries with real structured output — a small
example of not trusting free-form LLM output at the application boundary.

### Try the multi-agent system

On the Agent Trace page, try:

- `"What is the maximum supply of Bitcoin?"` → routes to `crypto_knowledge_agent`
- `"What is the current price of ethereum?"` → routes to `market_agent`
- `"ETH price, and when was the Merge?"` → calls **both**, combined into one answer
- `"I was charged twice for my subscription, please escalate this."` → drafts a ticket
  and waits for you to click **Approve & File** before it touches the real database

### Run the adversarial test

```bash
python prompt_injection_test.py
```

Reproduces both attacks from [`prompt_injection_test.md`](prompt_injection_test.md)
against the live escalation gate.

### Smoke test (no tokens spent)

```bash
python smoke_test.py
```

Starts the API on a free port, checks `/health` and `/docs`, shuts it down. Good as a
quick "did I break the server" check before spending API calls on anything else.

## Deploying to Render

Only `main.py` (the FastAPI backend) needs to go to Render — it serves both `/ask` and
`/agent`. The Streamlit UI stays local (or wherever you run it) and just points its API
base URL at your Render service; Streamlit itself isn't deployed here.

1. Push this repo to GitHub if it isn't remote-backed yet.
2. In the Render dashboard: **New → Web Service**, connect the repo.
3. **Root Directory:** `ai-engineering-bootcamp-v2/week-1v2`
4. **Build command:** `pip install --upgrade pip && pip install -r requirements.txt`
5. **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. **Environment variables:** add all six from the table above (Render's dashboard has
   an "Environment" tab — mark each as a secret, don't commit them).
7. Deploy, then confirm: `curl https://<your-service>.onrender.com/health`.
8. Ingest the knowledge base against the deployed index (skip if you're reusing an
   index you already ingested into locally):

   ```bash
   API_BASE_URL=https://<your-service>.onrender.com python ingest_corpus.py
   ```

9. Point the Streamlit app at it: run `streamlit run demo_page.py` locally, and in the
   sidebar set **API base URL** to `https://<your-service>.onrender.com` instead of
   `127.0.0.1:8000`. The Golden Set Eval and main pages read the same field; the Agent
   Trace page always runs `agent_app.py` in-process wherever Streamlit is running, so it
   needs its own `.env` with `GOOGLE_API_KEY` + Supabase vars regardless of where the API
   is deployed.

**If you want this reproducible as infrastructure-as-code** instead of clicking through
the dashboard, add a `render.yaml` in this folder (see
`multi-agent-systems/week-4/backend/render.yaml` in this repo for the pattern) with a
`services` entry using the build/start commands above and a `sync: false` env var per
secret.

**Known limitation:** Render's free tier spins down on idle, so the first request after
inactivity is slow (cold start). The `/agent` endpoint stacks a Gemini call, a CoinGecko
call, and possibly a Supabase MCP round-trip on top of that, so it will feel noticeably
slower than `/ask` on a cold instance — this is expected, not a bug.

## Troubleshooting

- `Cannot reach http://127.0.0.1:8000` — start the API server in another terminal.
- `OPENAI_API_KEY` error — make sure `.env` exists and contains a real key.
- `Address already in use` — another process is already on port `8000`; stop it or use
  a different port (and update the Streamlit sidebar URL to match).
- Streamlit opens but requests fail — confirm the sidebar API base URL matches where the
  API is actually running (local vs. Render).
- `/ask` always refuses / `/debug/retrieve` returns nothing — the knowledge base hasn't
  been ingested yet; run `python ingest_corpus.py` against the right `API_BASE_URL`.
- Agent Trace page errors on Gemini calls — check `GOOGLE_API_KEY` is set and that the
  model name in `agent_app.py` (`gemini-flash-latest`) still resolves; Gemini model
  names/aliases churn faster than most providers.
- Escalation ticket won't file even after approval — check `SUPABASE_ACCESS_TOKEN` /
  `SUPABASE_PROJECT_REF` are set and the token hasn't expired; drafting still works
  without them, only filing needs them.

## Instructor flow (original class version)

`main.py` and `demo_page.py`'s main page are the live student demo for the RAG-only
build. Open the stage files only to explain how the endpoint grew step by step:

| Stage | File | Teaching point |
|---|---|---|
| 1 | `stages/stage_1_bare_ask.py` | Smallest typed `/ask`: question in, string answer out. |
| 2 | `stages/stage_2_structured_output.py` | Add a Pydantic `Answer` schema and OpenAI structured output. |
| 3 | `stages/stage_3_guardrails_and_observability.py` | Add validation retry, model selection, latency, and cost. |

Run one stage at a time to teach the build-up live:

```bash
uvicorn stages.stage_1_bare_ask:app --host 127.0.0.1 --port 8000 --reload
uvicorn stages.stage_2_structured_output:app --host 127.0.0.1 --port 8000 --reload
uvicorn stages.stage_3_guardrails_and_observability:app --host 127.0.0.1 --port 8000 --reload
```

The multi-agent system (`agent_app.py`, `/agent`, and the Agent Trace / Golden Set Eval
pages) is the Session 3 capstone built on top of this, not part of the original
stage-by-stage teaching sequence.
