"""Golden-set eval page: run the fixed question set against the live /ask API,
grade retrieval/faithfulness/correctness, and log results locally so runs can be
compared over time — same pattern as week2's evaluation.py.
"""

import os

import streamlit as st

from golden_set import GOLDEN_SET, RESULTS_DIR, read_summary_rows, run_golden_set, write_report

st.set_page_config(page_title="Golden Set Eval", layout="centered")
st.title("Golden Set Eval")
st.caption(
    f"Runs {len(GOLDEN_SET)} fixed questions against the live API and checks retrieval hit, "
    "faithfulness (citation matches expectation), and correctness (LLM-judged against a known "
    "expected answer). Results are logged locally to results/ — nothing is written on Render."
)

base_url = st.sidebar.text_input(
    "API base URL", os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
)

with st.expander(f"Golden set questions ({len(GOLDEN_SET)})", expanded=False):
    for item in GOLDEN_SET:
        st.markdown(f"- **{item['question']}**")

note = st.text_input(
    "Note for this run",
    placeholder='e.g. "initial run", "changed chunking", "changed k"',
)
run_clicked = st.button("Run golden set eval", type="primary", disabled=not note.strip())
if not note.strip():
    st.caption("Add a note describing what changed before running — this is what makes runs comparable.")

if run_clicked:
    progress = st.progress(0.0, text="Starting...")
    completed = []

    def on_progress(result):
        completed.append(result)
        progress.progress(
            len(completed) / len(GOLDEN_SET),
            text=f"{len(completed)}/{len(GOLDEN_SET)}: {result['question'][:50]}",
        )

    try:
        results = run_golden_set(base_url, progress_callback=on_progress)
    except Exception as exc:
        st.error(f"Eval run failed: {exc}")
    else:
        progress.empty()
        run_file, summary_file, rates = write_report(results, note.strip())

        st.markdown("### This run")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Retrieval hit", f"{rates['retrieval_hit_rate']:.0%}")
        metric_cols[1].metric("Faithful", f"{rates['faithfulness_rate']:.0%}")
        metric_cols[2].metric("Correct", f"{rates['correctness_rate']:.0%}")

        for r in results:
            icon = "✅" if r["retrieval_hit"] and r["faithful"] and r["correct"] else "⚠️"
            with st.expander(f"{icon} {r['question']}", expanded=False):
                st.write(f"**Expected:** {r['expected_answer']}")
                st.write(f"**Actual:** {r['actual_answer']}")
                st.caption(
                    f"retrieval_hit={r['retrieval_hit']} | faithful={r['faithful']} | "
                    f"correct={r['correct']} — {r['correctness_reasoning']}"
                )

        st.success(f"Saved to `{run_file.relative_to(RESULTS_DIR.parent)}` and updated `summary.md`.")

st.divider()
st.markdown("### Run history")
summary_file = RESULTS_DIR / "summary.md"
summary_headers = ["Iteration", "Timestamp", "Retrieval Hit", "Faithful", "Correct", "Notes"]
history = read_summary_rows(summary_file, len(summary_headers))
if history:
    st.table([dict(zip(summary_headers, row)) for row in history])
else:
    st.caption("No runs logged yet.")
