"""Golden-set eval for the live /ask API: retrieval hit, faithfulness, correctness.

Calls the deployed API over HTTP (source of truth), then grades each answer with a
lightweight LLM judge — no RAGAS dependency, since this is a smaller, deterministic-
where-possible check rather than a full RAGAS scoring pass.
"""

from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

THIS_DIR = Path(__file__).resolve().parent
load_dotenv(THIS_DIR / ".env")

RESULTS_DIR = THIS_DIR / "results"
JUDGE_MODEL = "gpt-4o-mini"
_client: OpenAI | None = None

# Reused from the week2 golden set, plus one question the corpus does NOT cover,
# to prove refusal. expected_document_id is None for the refusal case.
GOLDEN_SET = [
    {
        "question": "What is Bitcoin?",
        "expected_answer": "Bitcoin is a decentralized digital currency that lets people send and "
        "receive value directly, without banks or a central authority.",
        "expected_document_id": "bitcoin_overview",
    },
    {
        "question": "What is the blockchain?",
        "expected_answer": "The blockchain is a public, distributed ledger that records every "
        "Bitcoin transaction; nodes each keep a copy, and transactions are grouped into "
        "cryptographically linked blocks.",
        "expected_document_id": "bitcoin_overview",
    },
    {
        "question": "How does Bitcoin mining work?",
        "expected_answer": "Miners compete to solve a Proof-of-Work puzzle; the first to solve it "
        "adds the next block and earns newly created bitcoin plus transaction fees.",
        "expected_document_id": "bitcoin_overview",
    },
    {
        "question": "What is the maximum supply of Bitcoin?",
        "expected_answer": "Bitcoin's supply is capped at 21 million coins.",
        "expected_document_id": "bitcoin_overview",
    },
    {
        "question": "What is the difference between a public key and a private key in a Bitcoin wallet?",
        "expected_answer": "A private key signs transactions and proves ownership of funds; the "
        "public key derives the wallet address others send funds to.",
        "expected_document_id": "bitcoin_overview",
    },
    {
        "question": "What is the capital of France?",
        "expected_answer": "I don't have enough information to answer that.",
        "expected_document_id": None,
    },
]


class CorrectnessJudgement(BaseModel):
    correct: bool
    reasoning: str


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def judge_correctness(question: str, expected_answer: str, actual_answer: str) -> CorrectnessJudgement:
    completion = get_client().chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Judge whether the actual answer correctly answers the question, using the "
                    "expected answer as the standard for what must be present. The expected answer "
                    "is a terse reference, not a word limit — the actual answer may include "
                    "additional correct, relevant detail without being marked incorrect. Only mark "
                    "it incorrect if it is missing the expected answer's key fact(s) or states "
                    "something that contradicts them.\n\n"
                    f"Question: {question}\n"
                    f"Expected answer: {expected_answer}\n"
                    f"Actual answer: {actual_answer}"
                ),
            }
        ],
        response_format=CorrectnessJudgement,
    )
    return completion.choices[0].message.parsed


def run_golden_set(api_base_url: str, progress_callback=None) -> list[dict]:
    """Calls the live /ask endpoint for every golden-set item and grades the result."""

    results = []
    for item in GOLDEN_SET:
        response = httpx.post(
            f"{api_base_url.rstrip('/')}/ask",
            json={"question": item["question"]},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()

        answer_text = data["answer"]["answer"]
        citations = data["answer"].get("citations", [])
        retrieved_doc_ids = {s["document_id"] for s in data.get("sources", [])}
        expected_doc_id = item["expected_document_id"]

        if expected_doc_id is None:
            # Refusal case: "retrieval hit" here means correctly refusing (no citations).
            retrieval_hit = True
            faithful = not citations
        else:
            retrieval_hit = expected_doc_id in retrieved_doc_ids
            faithful = expected_doc_id in citations

        judgement = judge_correctness(item["question"], item["expected_answer"], answer_text)

        result = {
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "actual_answer": answer_text,
            "retrieval_hit": retrieval_hit,
            "faithful": faithful,
            "correct": judgement.correct,
            "correctness_reasoning": judgement.reasoning,
        }
        results.append(result)
        if progress_callback:
            progress_callback(result)

    return results


def next_iteration() -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    existing = [int(p.stem.split("_")[1]) for p in RESULTS_DIR.glob("run_*.md")]
    return max(existing, default=0) + 1


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def format_row(cells):
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " |"

    lines = [format_row(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for row in rows:
        lines.append(format_row(row))
    return "\n".join(lines)


def read_summary_rows(summary_file: Path, num_columns: int) -> list[list[str]]:
    if not summary_file.exists():
        return []
    data_lines = summary_file.read_text(encoding="utf-8").splitlines()[2:]
    rows = []
    for line in data_lines:
        if not line.strip():
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|", num_columns - 1)]
        cells += [""] * (num_columns - len(cells))
        rows.append(cells)
    return rows


def write_report(results: list[dict], note: str) -> tuple[Path, Path, dict]:
    iteration = next_iteration()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total = len(results)
    retrieval_hits = sum(r["retrieval_hit"] for r in results)
    faithful_count = sum(r["faithful"] for r in results)
    correct_count = sum(r["correct"] for r in results)
    rates = {
        "retrieval_hit_rate": retrieval_hits / total,
        "faithfulness_rate": faithful_count / total,
        "correctness_rate": correct_count / total,
    }

    lines = [f"# Golden Set Run {iteration}", f"**Timestamp:** {timestamp}"]
    if note:
        lines.append(f"**Notes:** {note}")
    lines += ["", "## Questions & Answers", ""]
    for r in results:
        lines.append(f"**Q: {r['question']}**")
        lines.append("")
        lines.append(f"Expected: {r['expected_answer']}")
        lines.append("")
        lines.append(f"Actual: {r['actual_answer']}")
        lines.append("")
        lines.append(f"Correctness reasoning: {r['correctness_reasoning']}")
        lines.append("")

    lines.append("## Scores")
    lines.append("")
    headers = ["Question", "Retrieval Hit", "Faithful", "Correct"]
    score_rows = [
        [
            r["question"],
            "yes" if r["retrieval_hit"] else "no",
            "yes" if r["faithful"] else "no",
            "yes" if r["correct"] else "no",
        ]
        for r in results
    ]
    score_rows.append(
        [
            "**Rate**",
            f"**{rates['retrieval_hit_rate']:.0%}**",
            f"**{rates['faithfulness_rate']:.0%}**",
            f"**{rates['correctness_rate']:.0%}**",
        ]
    )
    lines.append(markdown_table(headers, score_rows))

    run_file = RESULTS_DIR / f"run_{iteration:03d}.md"
    run_file.write_text("\n".join(lines), encoding="utf-8")

    summary_file = RESULTS_DIR / "summary.md"
    summary_headers = ["Iteration", "Timestamp", "Retrieval Hit", "Faithful", "Correct", "Notes"]
    summary_rows = read_summary_rows(summary_file, len(summary_headers))
    summary_rows.append(
        [
            str(iteration),
            timestamp,
            f"{rates['retrieval_hit_rate']:.0%}",
            f"{rates['faithfulness_rate']:.0%}",
            f"{rates['correctness_rate']:.0%}",
            note,
        ]
    )
    summary_file.write_text(markdown_table(summary_headers, summary_rows) + "\n", encoding="utf-8")

    return run_file, summary_file, rates
