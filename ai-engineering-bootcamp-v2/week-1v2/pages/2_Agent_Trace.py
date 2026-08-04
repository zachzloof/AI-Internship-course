"""Live Think -> Act -> Observe trace for the Session 3 ADK agent (agent_app.py).

Run:
  streamlit run demo_page.py
  (this page is auto-discovered from pages/ -- pick "Agent Trace" in the sidebar)

Requires GOOGLE_API_KEY (Gemini) and PINECONE_API_KEY (the search_docs tool) in .env.
"""

import asyncio

import streamlit as st

from agent_app import MAX_LLM_CALLS, MODEL, root_agent, run_agent_steps

st.set_page_config(page_title="Agent Trace", layout="centered")
st.title("Crypto Knowledge Agent -- Live Trace")
st.caption(
    f"Runs the ADK agent (`{MODEL}`, capped at {MAX_LLM_CALLS} model calls) against your "
    "live Pinecone knowledge base and renders each Think / Act / Observe step as it happens."
)

with st.expander("What is this agent instructed to do?", expanded=False):
    st.text(root_agent.instruction)
    st.caption("Tool available: `search_docs` -- real semantic search, not mocked data.")

question = st.text_area(
    "Question",
    "By what percentage did Ethereum's energy consumption drop after Proof-of-Stake?",
    height=80,
)
run_clicked = st.button("Run agent", type="primary", disabled=not question.strip())

STEP_STYLE = {
    "act": ("🛠️", "ACT"),
    "observe": ("📥", "OBSERVE"),
    "think": ("💡", "THINK"),
}


def render_step(container, step_num: int, step: dict) -> None:
    icon, label = STEP_STYLE[step["type"]]
    with container:
        if step["type"] == "act":
            with st.expander(f"{icon} {label} {step_num} -- call `{step['tool']}`", expanded=True):
                st.json(step["args"])
        elif step["type"] == "observe":
            with st.expander(f"{icon} {label} {step_num} -- `{step['tool']}` result", expanded=True):
                st.json(step["result"])
        elif step["type"] == "think":
            st.markdown(f"{icon} **{label} {step_num}** -- model reached a final answer")


if run_clicked:
    log_container = st.container()

    async def drive():
        steps = []
        step_num = 0
        async for step in run_agent_steps(root_agent, question):
            step_num += 1
            steps.append(step)
            render_step(log_container, step_num, step)
        return steps

    try:
        with st.spinner("Agent is thinking..."):
            steps = asyncio.run(drive())
    except Exception as exc:
        st.error(f"Agent run failed: {exc}")
    else:
        final_think = next((s for s in reversed(steps) if s["type"] == "think"), None)
        st.divider()
        if final_think:
            st.markdown("### Final Answer")
            st.success(final_think["text"])
        else:
            st.warning(
                f"No final answer in {len(steps)} steps -- likely hit the "
                f"{MAX_LLM_CALLS}-call limit. Check the trace above."
            )
