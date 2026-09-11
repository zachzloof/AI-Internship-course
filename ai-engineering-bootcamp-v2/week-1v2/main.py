"""Week 1 v2 demo API: one compact `/ask` endpoint for the intro class.

Run:
  uvicorn main:app --host 127.0.0.1 --port 8000 --reload
"""

import json
import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

THIS_DIR = Path(__file__).resolve().parent
load_dotenv(THIS_DIR / ".env")
load_dotenv(THIS_DIR.parent / ".env")

from fastapi import FastAPI, HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langfuse import get_client, observe, propagate_attributes

# Langfuse must be imported (and its OpenAI patch applied) after env vars are loaded and
# before the OpenAI client is constructed, or it can't pick up credentials / patch the
# client in time -- see langfuse.com/docs "Common Mistakes" on import order.
from langfuse.openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, Field, ValidationError

langfuse = get_client()

app = FastAPI(title="Week 1 v2 /ask Demo")
_client: OpenAI | None = None
_pinecone_client: Pinecone | None = None
_pinecone_index = None

ModelName = Literal["gpt-4o-mini", "gpt-4o", "o3-mini"]
DEFAULT_MODEL: ModelName = "gpt-4o-mini"
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "o3-mini": (0.0011, 0.0044),
}

# Same embedding model must be used at ingest time and query time, or similarity
# scores are meaningless (vectors from different models aren't comparable).
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "week1v2-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

GROUNDING_PROMPT_TEMPLATE = (
    "Answer using ONLY the context below. Do not use outside or prior knowledge.\n"
    "If the context does not contain the answer, respond with exactly: "
    "\"I don't have enough information to answer that.\" and leave citations empty.\n\n"
    "Only use the parts of the context that directly answer the question — do not pad "
    "the answer with unrelated retrieved details just because they were returned. "
    "In the citations field, list ONLY the plain document_id value (e.g. \"bitcoin_overview\") "
    "of every chunk you actually used to write the answer — not the chunk number, not the "
    "\"[document_id: ... | chunk ...]\" label, just the id itself. Do not list chunks you were "
    "given but didn't use.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


class Answer(BaseModel):
    """The model output shape we want every caller to receive."""

    answer: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool
    citations: list[str] = Field(
        default_factory=list,
        description="document_id of each retrieved chunk actually used to answer, empty if none",
    )


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    model: ModelName | None = None
    force_bad: bool = False


class AttemptResult(BaseModel):
    attempt: int
    step: str
    ok: bool
    message: str
    raw_output: str | None = None
    validation_error: str | None = None


class RetrievedChunk(BaseModel):
    document_id: str
    chunk_index: int
    source: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: Answer
    tokens_used: int
    model: str
    latency_ms: int
    cost_usd: float
    attempts: list[AttemptResult]
    sources: list[RetrievedChunk] = []


class IngestRequest(BaseModel):
    text: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source: str | None = None


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    status: str


class AgentRequest(BaseModel):
    message: str = Field(min_length=1)


class AgentStep(BaseModel):
    """One tool call the agent made. `observation` is truncated -- this is a summary
    for API callers, not a full trace (see agent_loop_proof.md / the Streamlit Agent
    Trace page for the complete Think/Act/Observe log)."""

    tool: str
    observation: str


class AgentResponse(BaseModel):
    """Deliberately narrow: only `answer` and `steps` can ever be returned. Because
    FastAPI validates outgoing data against this model, there is no field this
    endpoint could accidentally leak an API key or env var through."""

    answer: str
    steps: list[AgentStep]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def get_pinecone_index():
    """Lazily create the Pinecone client/index, creating the index on first use."""

    global _pinecone_client, _pinecone_index
    if _pinecone_index is None:
        if not PINECONE_API_KEY:
            raise HTTPException(status_code=500, detail="PINECONE_API_KEY is not set")

        _pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        existing_names = {idx["name"] for idx in _pinecone_client.list_indexes()}
        if PINECONE_INDEX_NAME not in existing_names:
            _pinecone_client.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSIONS,
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )
        _pinecone_index = _pinecone_client.Index(PINECONE_INDEX_NAME)

    return _pinecone_index


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed one or more texts with the same model used at ingest and query time."""

    response = get_client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


@app.get("/health/pinecone")
def health_pinecone() -> dict:
    """Confirms Pinecone is reachable and returns basic index stats. Makes no LLM calls."""

    try:
        index = get_pinecone_index()
        stats = index.describe_index_stats()
        return {
            "status": "ok",
            "index_name": PINECONE_INDEX_NAME,
            "total_vectors": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", EMBEDDING_DIMENSIONS),
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Pinecone unreachable: {exc}")


def delete_existing_document_chunks(index, document_id: str) -> None:
    """Remove any chunks from a previous ingest of this document_id.

    Pinecone serverless indexes can't delete-by-metadata-filter, but IDs are
    "{document_id}::{chunk_index}", so listing by ID prefix finds them instead.
    Without this, re-ingesting a shorter version of a document leaves old trailing
    chunks stranded in the index.
    """

    prefix = f"{document_id}::"
    existing_ids = [item.id for batch in index.list(prefix=prefix) for item in batch]
    if existing_ids:
        index.delete(ids=existing_ids)


@app.post("/ingest")
@observe(name="ingest-document")
def ingest(body: IngestRequest) -> IngestResponse:
    """Chunk, embed, and upsert a document into the vector store.

    curl -s -X POST http://127.0.0.1:8000/ingest \
      -H "Content-Type: application/json" \
      -d '{"text": "Remote work: up to 3 days per week with manager approval.", "document_id": "handbook"}'
    """
    with propagate_attributes(tags=["ingest"]):
        langfuse.update_current_span(
            input={
                "document_id": body.document_id,
                "source": body.source,
                "text_length": len(body.text),
            }
        )

        text = body.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text must not be empty")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks produced from input text")

        source = body.source or body.document_id
        embeddings = embed_texts(chunks)
        vectors = [
            {
                "id": f"{body.document_id}::{i}",
                "values": embedding,
                "metadata": {
                    "document_id": body.document_id,
                    "chunk_index": i,
                    "source": source,
                    "text": chunk,
                },
            }
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        index = get_pinecone_index()
        delete_existing_document_chunks(index, body.document_id)
        index.upsert(vectors=vectors)

        result = IngestResponse(document_id=body.document_id, chunks_indexed=len(chunks), status="ok")
        langfuse.update_current_span(output=result.model_dump())
        return result


def retrieve_chunks(question: str, k: int) -> list[RetrievedChunk]:
    """Embed the question and return the top-k most similar chunks. No LLM call.

    Shared by /ask, /debug/retrieve, and agent_app.py's search_docs tool -- instrumented
    here, not at each call site, so every caller gets a properly typed `retriever`
    observation (distinct from the `generation` type the embedding call itself gets via
    the langfuse.openai wrapper) nested under whatever span is currently active.
    """

    with langfuse.start_as_current_observation(
        as_type="retriever", name="retrieve-chunks", input={"question": question, "k": k}
    ) as retriever_span:
        query_embedding = embed_texts([question])[0]
        index = get_pinecone_index()
        result = index.query(vector=query_embedding, top_k=k, include_metadata=True)

        chunks = [
            RetrievedChunk(
                document_id=match["metadata"].get("document_id", "unknown"),
                chunk_index=match["metadata"].get("chunk_index", -1),
                source=match["metadata"].get("source", "unknown"),
                score=match["score"],
                text=match["metadata"].get("text", ""),
            )
            for match in result["matches"]
        ]
        retriever_span.update(
            output=[
                {"document_id": c.document_id, "chunk_index": c.chunk_index, "score": c.score}
                for c in chunks
            ]
        )
        return chunks


def build_grounded_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[document_id: {c.document_id} | chunk {c.chunk_index}]\n{c.text}" for c in chunks
    )
    return GROUNDING_PROMPT_TEMPLATE.format(context=context, question=question)


def normalize_citations(citations: list[str], known_document_ids: set[str]) -> list[str]:
    """The model sometimes echoes the full "[document_id: X | chunk N]" context label instead
    of the plain id. Match each raw citation against known ids by substring rather than trusting
    exact formatting, so the field stays reliable regardless of minor prompt-following slips.
    """

    matched = []
    for raw in citations:
        for doc_id in known_document_ids:
            if doc_id in raw and doc_id not in matched:
                matched.append(doc_id)
    return matched


@app.get("/debug/retrieve")
def debug_retrieve(q: str, k: int = 5) -> list[RetrievedChunk]:
    """Retrieval only, no generation — verify the right chunks come back before wiring /ask.

    curl -s "http://127.0.0.1:8000/debug/retrieve?q=What+is+the+remote+work+policy%3F"
    """

    if not q.strip():
        raise HTTPException(status_code=400, detail="q must not be empty")
    # No generation happens here, so retrieve_chunks' own `retriever` observation is
    # already the whole unit of work -- an extra wrapping span would just be a redundant
    # parent with nothing else under it. propagate_attributes only adds the tag.
    with propagate_attributes(tags=["debug"]):
        return retrieve_chunks(q, k)


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    input_per_1k, output_per_1k = MODEL_PRICES_PER_1K.get(
        model, MODEL_PRICES_PER_1K[DEFAULT_MODEL]
    )
    return (prompt_tokens / 1000 * input_per_1k) + (
        completion_tokens / 1000 * output_per_1k
    )


def usage_counts(completion) -> tuple[int, int, int]:
    usage = completion.usage
    if usage is None:
        return 0, 0, 0
    return usage.total_tokens, usage.prompt_tokens, usage.completion_tokens


def call_structured_model(question: str, model: ModelName) -> tuple[Answer, int, int, int]:
    completion = get_client().chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": question}],
        response_format=Answer,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    total_tokens, prompt_tokens, completion_tokens = usage_counts(completion)
    return parsed, total_tokens, prompt_tokens, completion_tokens


def call_malformed_json_once(question: str, model: ModelName) -> tuple[str, int, int, int]:
    """Demo-only path: force one malformed response so students can see retry."""

    completion = get_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{question}\n\n"
                    "Reply with ONLY JSON using keys answer, confidence, sources_needed. "
                    "Set confidence to the string 'very high' instead of a number."
                ),
            }
        ],
    )

    raw = completion.choices[0].message.content or ""
    total_tokens, prompt_tokens, completion_tokens = usage_counts(completion)
    return raw, total_tokens, prompt_tokens, completion_tokens


@app.post("/ask")
@observe(name="ask-request")
def ask(body: AskRequest) -> AskResponse:
    with propagate_attributes(tags=["ask"]):
        model = body.model or DEFAULT_MODEL
        langfuse.update_current_span(input={"question": body.question, "model": model})
        last_error: str | None = None
        attempts: list[AttemptResult] = []
        total_tokens_used = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        start = time.perf_counter()

        retrieved = retrieve_chunks(body.question, RAG_TOP_K)

        # Genuinely empty index (nothing ingested yet) — refuse without spending an LLM call.
        # A non-empty-but-irrelevant retrieval still goes to the model, which judges relevance
        # itself via the grounding prompt (similarity search always returns *something*).
        if not retrieved:
            result = AskResponse(
                answer=Answer(
                    answer="I don't have enough information to answer that.",
                    confidence=0.0,
                    sources_needed=True,
                ),
                tokens_used=0,
                model=model,
                latency_ms=int((time.perf_counter() - start) * 1000),
                cost_usd=0.0,
                attempts=[],
                sources=[],
            )
            langfuse.update_current_span(output=result.answer.model_dump())
            return result

        grounded_prompt = build_grounded_prompt(body.question, retrieved)

        for attempt in range(2):
            try:
                if body.force_bad and attempt == 0:
                    raw, tokens_used, prompt_tokens, completion_tokens = call_malformed_json_once(
                        body.question, model
                    )
                    total_tokens_used += tokens_used
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens

                    try:
                        answer = Answer.model_validate_json(raw)
                    except ValidationError as exc:
                        last_error = str(exc)
                        attempts.append(
                            AttemptResult(
                                attempt=attempt + 1,
                                step="forced_bad_json",
                                ok=False,
                                message="Validation failed, so the endpoint retries with structured output.",
                                raw_output=raw,
                                validation_error=str(exc),
                            )
                        )
                        continue

                    attempts.append(
                        AttemptResult(
                            attempt=attempt + 1,
                            step="forced_bad_json",
                            ok=True,
                            message="Unexpectedly passed validation.",
                            raw_output=raw,
                        )
                    )
                else:
                    answer, tokens_used, prompt_tokens, completion_tokens = call_structured_model(
                        grounded_prompt, model
                    )
                    answer.citations = normalize_citations(
                        answer.citations, {c.document_id for c in retrieved}
                    )
                    total_tokens_used += tokens_used
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    attempts.append(
                        AttemptResult(
                            attempt=attempt + 1,
                            step="structured_output",
                            ok=True,
                            message="Structured output matched the Answer schema.",
                        )
                    )

                latency_ms = int((time.perf_counter() - start) * 1000)
                cost_usd = compute_cost_usd(
                    model, total_prompt_tokens, total_completion_tokens
                )
                result = AskResponse(
                    answer=answer,
                    tokens_used=total_tokens_used,
                    model=model,
                    latency_ms=latency_ms,
                    cost_usd=round(cost_usd, 6),
                    attempts=attempts,
                    sources=retrieved,
                )
                langfuse.update_current_span(
                    output={
                        "answer": answer.answer,
                        "confidence": answer.confidence,
                        "citations": answer.citations,
                        "sources_needed": answer.sources_needed,
                        "tokens_used": total_tokens_used,
                        "cost_usd": result.cost_usd,
                    }
                )
                return result
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)
                attempts.append(
                    AttemptResult(
                        attempt=attempt + 1,
                        step="structured_output",
                        ok=False,
                        message="Structured output failed validation.",
                        validation_error=str(exc),
                    )
                )

        raise HTTPException(
            status_code=502,
            detail=f"Model response failed schema validation after retry: {last_error}",
        )


MAX_OBSERVATION_CHARS = 300


@app.post("/agent")
async def agent(body: AgentRequest) -> AgentResponse:
    """Runs the Session 3 ADK agent (crypto_knowledge_agent) on a user message.

    Returns the final answer plus a short steps[] summary (tool name + truncated
    observation per tool call) -- not the full Think/Act/Observe trace. Response is
    validated against AgentResponse, so nothing beyond those two fields can ever be
    returned; no keys, no env vars.

    agent_app is imported lazily (inside the function, not at module load) because
    agent_app.py itself imports retrieve_chunks from this module -- importing it at
    the top of main.py would create a circular import.

    curl -s -X POST http://127.0.0.1:8000/agent \
      -H "Content-Type: application/json" \
      -d '{"message": "What is the maximum supply of Bitcoin?"}'

    curl -s -X POST https://your-service.onrender.com/agent \
      -H "Content-Type: application/json" \
      -d '{"message": "What is the maximum supply of Bitcoin?"}'
    """
    from agent_app import root_agent, run_agent_steps

    steps: list[AgentStep] = []
    answer = "(no response)"

    # Not wrapped in @observe: GoogleADKInstrumentor (instrumented once in agent_app.py,
    # which every consumer of run_agent_steps goes through) already gives runner.run_async
    # its own properly typed root trace -- an extra span here would just double up on it,
    # not add anything (see the skill's "don't emit duplicate dispatch + execution nodes").
    # trace_tags only labels that existing trace as having come through the HTTP API.
    try:
        async for step in run_agent_steps(root_agent, body.message, trace_tags=["agent-api"]):
            if step["type"] == "observe":
                observation = json.dumps(step["result"])[:MAX_OBSERVATION_CHARS]
                steps.append(AgentStep(tool=step["tool"], observation=observation))
            elif step["type"] == "think":
                answer = step["text"]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent run failed: {exc}")

    return AgentResponse(answer=answer, steps=steps)
