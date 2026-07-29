import sys
import types
from datetime import datetime
from pathlib import Path

# ragas 0.4.3 unconditionally imports langchain_community.chat_models.vertexai (only for an
# internal isinstance check), but that submodule was dropped in langchain-community 0.4.x.
# Stub it out rather than pin an older, incompatible langchain-community version.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

from langchain_core.messages import AIMessage, ToolMessage

from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from config import embeddings, chat_model
from ask import rag_agent

RESULTS_DIR = Path(__file__).resolve().parent / "results"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# Golden set: questions + human-written ground-truth answers, used as the
# fixed benchmark to measure the RAG pipeline against.
golden_set = [
    {
        "question": "What is Bitcoin?",
        "reference": "Bitcoin is a decentralized digital currency, introduced in 2008 by Satoshi Nakamoto, "
        "that lets people send and receive value directly without banks or a central authority.",
    },
    {
        "question": "What is the blockchain?",
        "reference": "The blockchain is a public, distributed ledger that records every Bitcoin transaction. "
        "Thousands of nodes each keep a copy, and transactions are grouped into cryptographically "
        "linked blocks that form a chain.",
    },
    {
        "question": "How does Bitcoin mining work?",
        "reference": "Mining is the process where miners compete to solve a difficult Proof-of-Work puzzle. "
        "The first miner to solve it adds the next block to the chain and earns newly created bitcoin "
        "plus transaction fees.",
    },
    {
        "question": "What is the maximum supply of Bitcoin?",
        "reference": "Bitcoin's supply is capped at 21 million coins, enforced by the protocol, with the "
        "issuance rate cut in half roughly every four years in an event called the halving.",
    },
    {
        "question": "What is the difference between a public key and a private key in a Bitcoin wallet?",
        "reference": "A private key is a secret used to digitally sign transactions and prove ownership of "
        "funds, while the public key is used to derive the public wallet address that others send "
        "funds to.",
    },
]

def run_agent(question):
    """Drive the real rag_agent from ask.py and pull out what it actually retrieved and answered."""
    contexts = []
    answer = ""
    for event in rag_agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        last_message = event["messages"][-1]
        if isinstance(last_message, ToolMessage) and last_message.artifact:
            contexts = [doc.page_content for doc in last_message.artifact]
        elif isinstance(last_message, AIMessage) and last_message.content:
            answer = last_message.content
    return contexts, answer


def build_eval_rows():
    eval_rows = []
    for item in golden_set:
        contexts, answer = run_agent(item["question"])

        eval_rows.append({
            "user_input": item["question"],
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": item["reference"],
        })
        print(f"{item['question']}\n   -> {answer[:120]}{'...' if len(answer) > 120 else ''}\n")
    return eval_rows


def next_iteration():
    RESULTS_DIR.mkdir(exist_ok=True)
    existing = [int(p.stem.split("_")[1]) for p in RESULTS_DIR.glob("run_*.md")]
    return max(existing, default=0) + 1


def markdown_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def format_row(cells):
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " |"

    lines = [
        format_row(headers),
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    for row in rows:
        lines.append(format_row(row))
    return "\n".join(lines)


def read_summary_rows(summary_file, num_columns):
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


def write_report(iteration, timestamp, eval_rows, df, note=""):
    lines = [f"# Evaluation Run {iteration}", f"**Timestamp:** {timestamp}"]
    if note:
        lines.append(f"**Notes:** {note}")
    lines += ["", "## Questions & Answers", ""]
    for row in eval_rows:
        lines.append(f"**Q: {row['user_input']}**")
        lines.append("")
        lines.append(f"A: {row['response']}")
        lines.append("")

    lines.append("## Scores")
    lines.append("")
    headers = ["Question"] + [m.replace("_", " ").title() for m in METRICS]
    score_rows = [
        [row["user_input"]] + [f"{row[m]:.2f}" for m in METRICS]
        for _, row in df.iterrows()
    ]
    averages = df[METRICS].mean()
    score_rows.append(["**Average**"] + [f"**{averages[m]:.2f}**" for m in METRICS])
    lines.append(markdown_table(headers, score_rows))

    run_file = RESULTS_DIR / f"run_{iteration:03d}.md"
    run_file.write_text("\n".join(lines), encoding="utf-8")

    summary_file = RESULTS_DIR / "summary.md"
    summary_headers = ["Iteration", "Timestamp"] + [m.replace("_", " ").title() for m in METRICS] + ["Notes"]
    summary_rows = read_summary_rows(summary_file, len(summary_headers))
    summary_rows.append([str(iteration), timestamp] + [f"{averages[m]:.2f}" for m in METRICS] + [note])
    summary_file.write_text(markdown_table(summary_headers, summary_rows) + "\n", encoding="utf-8")

    return run_file, summary_file


def main(note=""):
    print(f"Golden set ready: {len(golden_set)} questions with ground-truth answers\n")

    eval_rows = build_eval_rows()
    print(f"Collected {len(eval_rows)} evaluation rows (question + contexts + answer + reference)\n")

    eval_dataset = EvaluationDataset.from_list(eval_rows)

    # RAGAS uses an LLM as the "judge" — here gpt-4o-mini grades gpt-4o-mini's own answers.
    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=LangchainLLMWrapper(chat_model),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
    )

    print("Average scores across the golden set:")
    print(result)

    df = result.to_pandas()
    table = df[["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]]
    print("\nPer-question scores:")
    print(table.to_string(index=False))

    iteration = next_iteration()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_file, summary_file = write_report(iteration, timestamp, eval_rows, df, note=note)
    print(f"\nSaved run {iteration} to {run_file}")
    print(f"Updated comparison summary at {summary_file}")


if __name__ == "__main__":
    main(note=" ".join(sys.argv[1:]))
