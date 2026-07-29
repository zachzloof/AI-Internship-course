import shutil

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma

from config import DATA_DIR, PERSIST_DIR, COLLECTION_NAME, embeddings

SOURCE_FILE = DATA_DIR / "bitcoin_overview.txt"


def main():
    text = SOURCE_FILE.read_text(encoding="utf-8")

    # Split on markdown headers rather than raw character count, so each chunk is one
    # complete section and the header travels as metadata instead of sometimes ending
    # up stranded as its own near-empty chunk.
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "title"), ("##", "section")])
    chunks = splitter.split_text(text)
    for chunk in chunks:
        chunk.metadata["source"] = str(SOURCE_FILE)
    print(f"Split {SOURCE_FILE.name} into {len(chunks)} section-based chunks")

    # Chroma.from_documents appends to an existing persist_directory, so start clean
    # on every run to avoid storing (and retrieving) duplicate chunks.
    shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION_NAME,
    )
    print(f"Vector store saved to {PERSIST_DIR}")


if __name__ == "__main__":
    main()
