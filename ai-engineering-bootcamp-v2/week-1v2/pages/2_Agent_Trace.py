"""Live Think -> Act -> Observe trace for the crypto triage router (agent_app.py).

Run:
  streamlit run demo_page.py
  (this page is auto-discovered from pages/ -- pick "Agent Trace" in the sidebar)

Requires GOOGLE_API_KEY (Gemini), PINECONE_API_KEY (search_docs), and Node/npx on PATH
(the escalation ticket tool runs an MCP filesystem server as a subprocess) in .env.
"""

import asyncio

import streamlit as st
from google.adk.sessions import InMemorySessionService

from agent_app import MAX_LLM_CALLS, MODEL, new_session_id, root_agent, run_agent_steps

st.set_page_config(page_title="Agent Trace", layout="centered")
st.title("Crypto Triage Router -- Live Trace")
st.caption(
    f"Runs the ADK router (`{MODEL}`, capped at {MAX_LLM_CALLS} model calls/turn) and renders "
    "each Think / Act / Observe step as it happens. Routes to whichever specialist fits: "
    "knowledge (real Pinecone RAG), market data (real CoinGecko API), or escalation "
    "(drafts a ticket, then waits for your approval below before filing it for real)."
)

with st.expander("What can this system do?", expanded=False):
    st.markdown("**Router** decides which specialist handles the message:")
    st.text(root_agent.instruction)
    for sub in root_agent.sub_agents:
        st.markdown(f"**{sub.name}**")
        st.caption(sub.description)

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
            with st.expander(
                f"{icon} {label} {step_num} -- {step['author']} calls `{step['tool']}`",
                expanded=True,
            ):
                st.json(step["args"])
        elif step["type"] == "observe":
            with st.expander(
                f"{icon} {label} {step_num} -- `{step['tool']}` result", expanded=True
            ):
                st.json(step["result"])
        elif step["type"] == "think":
            st.markdown(f"{icon} **{label} {step_num}** -- {step['author']} reached a final answer")


def find_pending_ticket(steps: list[dict]) -> dict | None:
    """A ticket is pending approval if draft_ticket produced one and no write_file
    call in these same steps already filed it."""
    drafted = None
    for step in steps:
        if step["type"] == "observe" and step["tool"] == "draft_ticket":
            result = step["result"]
            if isinstance(result, dict) and result.get("status") == "drafted":
                drafted = result
        if step["type"] == "observe" and step["tool"] == "write_file":
            drafted = None  # already filed in this same run
    return drafted


async def run_turn(message: str, session_service, session_id, state_delta=None):
    log_container = st.container()
    steps = []
    step_num = 0
    async for step in run_agent_steps(
        root_agent,
        message,
        session_service=session_service,
        session_id=session_id,
        state_delta=state_delta,
    ):
        step_num += 1
        steps.append(step)
        render_step(log_container, step_num, step)
    return steps


if run_clicked:
    # Each new question starts a fresh session -- any previous unapproved draft is
    # abandoned, matching how a real support conversation would move on.
    service = InMemorySessionService()
    session_id = asyncio.run(new_session_id(service))
    st.session_state.pop("pending_ticket", None)

    try:
        with st.spinner("Agent is thinking..."):
            steps = asyncio.run(run_turn(question, service, session_id))
    except Exception as exc:
        st.error(f"Agent run failed: {exc}")
    else:
        final_think = next((s for s in reversed(steps) if s["type"] == "think"), None)
        st.divider()
        if final_think:
            st.markdown("### Answer")
            st.success(final_think["text"])
        else:
            st.warning(
                f"No final answer in {len(steps)} steps -- likely hit the "
                f"{MAX_LLM_CALLS}-call limit. Check the trace above."
            )

        pending = find_pending_ticket(steps)
        if pending:
            # Session state must persist across the rerun the Approve/Reject button
            # triggers -- Streamlit reruns the whole script on every click, so the
            # service/session_id from *this* run have to be stashed here to be
            # reused by the follow-up turn instead of starting a new session.
            st.session_state.pending_ticket = {
                "ticket_id": pending["ticket_id"],
                "ticket": pending["ticket"],
                "service": service,
                "session_id": session_id,
            }


if "pending_ticket" in st.session_state:
    pt = st.session_state.pending_ticket
    st.divider()
    st.warning(f"⏸️ **Escalation ticket `{pt['ticket_id']}` is drafted but NOT filed.**")
    st.json(pt["ticket"])
    st.caption(
        "The write_file tool is structurally blocked until you approve -- the model "
        "cannot file this itself, no matter what it or any retrieved content says."
    )
    col1, col2 = st.columns(2)
    approve_clicked = col1.button("✅ Approve & File", type="primary")
    reject_clicked = col2.button("❌ Reject")

    if approve_clicked:
        st.markdown("### Filing ticket...")
        try:
            asyncio.run(
                run_turn(
                    "The human has approved this escalation. Please file the ticket now.",
                    pt["service"],
                    pt["session_id"],
                    state_delta={"escalation_approved": True},
                )
            )
        except Exception as exc:
            st.error(f"Filing failed: {exc}")
        else:
            st.success(f"Ticket `{pt['ticket_id']}` filed.")
        del st.session_state.pending_ticket

    if reject_clicked:
        st.info(f"Ticket `{pt['ticket_id']}` rejected -- it was never filed.")
        del st.session_state.pending_ticket
