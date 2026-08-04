"""
Crypto Knowledge Agent -- single ADK agent with one real tool (Session 3 assignment)

Pattern copied from:
  ai-engineering-bootcamp/adk-multi-agent-systems/demo1_routing.py
    - single Agent(...) with tools=[...]
    - asyncio Runner + InMemorySessionService loop (the `ask()` / `main()` shape)

This is deliberately ONE agent, not a router: today's job is "answer a crypto question
using the knowledge base," which is a single multi-step task (decide to look something
up -> call the tool -> answer), not a choice between multiple specialists. The router
(conceptual/technical vs. market-data vs. account-specific, per the capstone triage plan)
only gets added once there's a second distinct specialist for it to route to -- e.g. a
market-data agent.

The tool (search_docs) is real: it calls Session 2's retrieve_chunks() against the same
live Pinecone index /ask uses, not a stub.

Run:
  python agent_app.py
"""

import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent, RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from main import retrieve_chunks, RAG_TOP_K

load_dotenv()

# NOTE: the ADK sample repo hardcodes "gemini-2.5-flash", which is deprecated for new
# API keys. Its usual pinned replacements (gemini-2.0-flash, gemini-2.5-flash-lite,
# even gemini-3.5-flash under momentary load) all 404'd or 503'd against this key when
# actually run. "gemini-flash-latest" is an alias Google keeps pointed at whatever
# current flash-tier model is live -- confirmed working end-to-end, including a real
# search_docs tool call, on 2026-08-04.
MODEL = "gemini-flash-latest"
MAX_LLM_CALLS = 6  # hard ceiling on model calls per run -- stops a runaway tool-call loop

# --- Tool (real: reuses Session 2's retrieve_chunks() against the live Pinecone index) ---


def search_docs(query: str) -> dict:
    """Search the ingested crypto knowledge base for chunks relevant to the query.

    Real semantic search against the live Pinecone index used by /ask and
    /debug/retrieve in main.py -- same embedding model, same data, no mocking.
    Returns the top matching chunks (document_id, similarity score, text), an
    empty "results" list if nothing matched, or an "error" key if the knowledge
    base could not be reached (e.g. missing Pinecone credentials) so the model
    can see the failure instead of the process crashing.
    """
    try:
        chunks = retrieve_chunks(query, RAG_TOP_K)
    except Exception as exc:
        return {"error": f"search_docs failed: {exc}"}

    if not chunks:
        return {"results": [], "note": "No matching chunks found in the knowledge base."}

    return {
        "results": [
            {"document_id": c.document_id, "score": round(c.score, 4), "text": c.text}
            for c in chunks
        ]
    }


# --- Agent ---

crypto_knowledge_agent = Agent(
    name="crypto_knowledge_agent",
    model=MODEL,
    description="Answers crypto/DeFi questions by searching the capstone knowledge base.",
    instruction=(
        "You are a crypto knowledge specialist for a support triage system.\n"
        "Goal: answer the user's crypto/DeFi question using ONLY information returned "
        "by search_docs -- never answer from your own prior knowledge.\n"
        "Constraints:\n"
        "- Always call search_docs at least once before answering.\n"
        "- Do not call the tool more than twice for a single question.\n"
        "- If search_docs returns an error or no results, say so explicitly instead of guessing.\n"
        "Done when: you have produced one final answer that is either grounded in a "
        "tool result or an explicit 'I don't have enough information' refusal."
    ),
    tools=[search_docs],
)

root_agent = crypto_knowledge_agent  # ADK convention: entrypoint/tools look for `root_agent`

# --- Runner: one shared event loop, two consumers (CLI printing, Streamlit UI) ---


async def run_agent_steps(agent: Agent, message: str, max_llm_calls: int = MAX_LLM_CALLS):
    """Runs the agent and yields Think/Act/Observe steps as they happen.

    Single source of truth for how ADK's event stream maps to Think/Act/Observe --
    both the CLI (`ask`, below) and the Streamlit trace page consume this instead of
    each re-implementing the runner loop.
    """
    service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="capstone", session_service=service)
    session = await service.create_session(app_name="capstone", user_id="user1")
    content = types.Content(role="user", parts=[types.Part(text=message)])
    run_config = RunConfig(max_llm_calls=max_llm_calls)

    async for event in runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=content,
        run_config=run_config,
    ):
        for call in event.get_function_calls():
            yield {"type": "act", "author": event.author, "tool": call.name, "args": call.args}
        for resp in event.get_function_responses():
            yield {
                "type": "observe",
                "author": event.author,
                "tool": resp.name,
                "result": resp.response,
            }
        if event.is_final_response() and event.content and event.content.parts:
            yield {"type": "think", "author": event.author, "text": event.content.parts[0].text}


async def ask(agent: Agent, message: str, max_llm_calls: int = MAX_LLM_CALLS) -> str:
    final_text = "(no response)"
    async for step in run_agent_steps(agent, message, max_llm_calls):
        if step["type"] == "act":
            print(f"[ACT]     {step['author']} -> {step['tool']}({step['args']})")
        elif step["type"] == "observe":
            print(f"[OBSERVE] {step['author']} <- {step['tool']}: {step['result']}")
        elif step["type"] == "think":
            print(f"[THINK]   {step['author']} reached a final answer")
            final_text = step["text"]
    return final_text


async def main():
    tests = [
        "What is the maximum supply of Bitcoin?",
        "How does a stablecoin stay pegged to $1?",
    ]
    for query in tests:
        print(f"\n--- User: {query} ---")
        answer = await ask(root_agent, query)
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
