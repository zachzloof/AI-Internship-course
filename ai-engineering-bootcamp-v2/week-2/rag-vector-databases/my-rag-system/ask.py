from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent
from langchain.tools import tool

from config import PERSIST_DIR, COLLECTION_NAME, embeddings, chat_model

vectorstore = Chroma(
    persist_directory=str(PERSIST_DIR),
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information about Bitcoin from the document store to help answer a query."""
    retrieved_docs = vectorstore.similarity_search(query, k=3)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


system_prompt = (
    "You are a Bitcoin Q&A assistant. You have access to a tool that retrieves context from a "
    "document about Bitcoin — always use it before answering.\n\n"
    "Answer using ONLY information found in the retrieved context. Do not use outside or prior "
    "knowledge, and do not state anything the context doesn't support. If the retrieved context "
    "doesn't contain the answer, say so plainly instead of guessing.\n\n"
    "The retrieval tool may return multiple chunks, but not all of them will be needed — only use "
    "the parts that directly answer the specific question asked. Do not summarize or work in every "
    "retrieved chunk just because it was returned. For a 'what is X' question, give the definition "
    "only; do not add why it matters, history, or related mechanics unless the question asks for "
    "them. Keep answers as short as fully answering the question allows.\n\n"
    "Always cite your sources."
)

rag_agent = create_agent(chat_model, [retrieve_context], system_prompt=system_prompt)


def ask(question: str):
    print(f"\n{'=' * 70}\nQuestion: {question}\n{'=' * 70}")
    for event in rag_agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        last_message = event["messages"][-1]

        if isinstance(last_message, HumanMessage):
            continue  # already printed as "Question" above

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print("Tool Calls:")
            for tool_call in last_message.tool_calls:
                print(f"  - {tool_call.get('name', 'unknown')}: {tool_call.get('args', {})}")
            print()
            continue

        if isinstance(last_message, ToolMessage):
            print(f"Chunks Retrieved:\n{last_message.content}\n")
            continue

        if last_message.content:
            print(f"Answer:\n{last_message.content}\n")


if __name__ == "__main__":
    print("Ask questions about Bitcoin (type 'quit' to exit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if question:
            ask(question)
