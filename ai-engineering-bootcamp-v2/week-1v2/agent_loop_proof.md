# Step 4: Prove the Loop

**Stack:** ADK

**Agent or workflow?** This is an agent because the model decides *whether and how* to call
`search_docs` — it writes its own query, decides after seeing the result whether one call was
enough or a second is needed, and decides whether to answer or refuse — rather than following a
fixed call-tool-then-format pipeline with no branching.

## Task run

Query chosen specifically because it isn't reliably answerable from memory alone: it asks for two
precise figures (an exact percentage and an exact technical term) that only a retrieval hit
against this corpus can guarantee, rather than a fact a general-purpose model might already know
approximately.

**Query:** "By what percentage did Ethereum's energy consumption drop after switching to
Proof-of-Stake, and what is the penalty called when a dishonest validator's staked Ether is
destroyed?"

**Run command:** `python agent_app.py` (this exact query run via the same `ask()` entrypoint)

## Log — Think → Act → Observe

```
--- User: By what percentage did Ethereum's energy consumption drop after switching to
Proof-of-Stake, and what is the penalty called when a dishonest validator's staked Ether
is destroyed? ---

[ACT]     crypto_knowledge_agent -> search_docs({'query': 'Ethereum energy consumption drop
percentage Proof-of-Stake validator penalty destroyed staked Ether slashing'})

[OBSERVE] crypto_knowledge_agent <- search_docs: {'results': [
    {'document_id': 'ethereum_smart_contracts', 'score': 0.6106,
     'text': '## Proof-of-Stake\n\n...validators who attempt to cheat the network can have
     a portion of their staked Ether destroyed, a penalty called "slashing." This change
     reduced Ethereum's energy consumption by more than 99 percent compared to
     Proof-of-Work.\n\n## Why It Matters'},
    {'document_id': 'ethereum_smart_contracts', 'score': 0.3669, ...},   # gas
    {'document_id': 'ethereum_smart_contracts', 'score': 0.3578, ...},   # tokens/dApps
    {'document_id': 'bitcoin_overview',         'score': 0.3261, ...},   # mining/PoW
    {'document_id': 'ethereum_smart_contracts', 'score': 0.2929, ...},   # EVM
]}

[THINK]   crypto_knowledge_agent reached a final answer

Agent: Based on the knowledge base:

* **Energy Consumption Drop:** Switching to Proof-of-Stake reduced Ethereum's energy
  consumption by **more than 99 percent** compared to Proof-of-Work.
* **Penalty Name:** The penalty in which a dishonest validator's staked Ether is destroyed
  is called **"slashing."**
```

No secrets in this log — `search_docs` returns only document text/scores from the ingested
corpus, and neither the Pinecone nor Gemini API keys ever appear in tool output or model output.

## What success looked like, mapped to what happened

- **Think:** the model read the question and proposed a `search_docs` call with its own
  synthesized query (not the raw question — it pulled out the key terms itself).
- **Act / Observe:** `search_docs` ran for real against the live Pinecone index and returned 5
  actual scored chunks, including the one containing both requested facts.
- **Final answer:** the model used the top result's exact figures ("more than 99 percent",
  "slashing") rather than paraphrasing from its own training knowledge.
