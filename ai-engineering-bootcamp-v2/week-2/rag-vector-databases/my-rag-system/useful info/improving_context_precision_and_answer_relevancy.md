# Improving Low RAGAS Scores: Context Precision & Answer Relevancy

## Fixing low `context_precision`

`context_precision` asks: *of the chunks I retrieved, how many were actually needed to answer the question?* A low score means the LLM's prompt got padded with chunks that were topically nearby but not actually useful — noise crowding out signal.

Look at the data first: for "What is Bitcoin?" and "What is the blockchain?", precision was ~0.58 (worst in the set), while "How does mining work?" and "max supply?" hit 1.00. That's a clue — something about those two questions' retrieval specifically is pulling in a weak 3rd chunk.

**1. Decrease K** — `k=3` is set in `retrieve_context` (`ask.py`) and the retrieval loop in `evaluation.py`. Every query always pulls exactly 3 chunks, even if only 1–2 are genuinely relevant. Forcing a fixed count means on easy/short questions you're guaranteed to drag in a marginal 3rd chunk just to fill the quota. Dropping to `k=2` removes that weakest chunk — precision goes up almost mechanically. The tradeoff is `context_recall`: if the true answer sometimes lives in that 3rd slot for other questions, you'll start missing it. That's why you change one knob and re-run `evaluation.py` to see the net effect, rather than assuming it's a free win.

**2. Improve chunking** — this is the one the retrieved-chunks trace already hinted at. In the "what is the blockchain?" trace, one of the 3 chunks retrieved was literally just `## The Blockchain: A Shared Public Ledger` with almost no body text. That happened because `ingest.py`'s splitter (`chunk_size=500, chunk_overlap=100`) sometimes cuts right after a markdown header, leaving a near-empty chunk that still "counts" as one of your 3 retrieval slots but contributes nothing. Fixing the splitter so headers stay attached to their paragraph (bigger chunk size, or a markdown-aware splitter that treats `## Heading` as metadata instead of inline content) stops you from wasting a retrieval slot on a content-free fragment.

**3. Metadata filtering** — the document is already organized into clean `## Section` chunks. If each chunk carried a `section` metadata field (like the notebook's `MarkdownHeaderTextSplitter` demo does), you could restrict `similarity_search` to only chunks whose section is plausibly relevant, instead of relying purely on embedding similarity across the whole document. This matters more as a document/corpus grows — with one short file it helps less, but it's the mechanism for when precision problems come from "right document, wrong section" rather than "right section, noisy chunk boundaries."

## Fixing low `answer_relevancy`

`answer_relevancy` asks: *does the generated answer actually address what was asked* — not whether it's true (that's `faithfulness`), but whether it's on-topic and focused. RAGAS estimates this by generating hypothetical questions from the answer and checking how close they land to the actual question.

**1. Improve the prompt template** — the current `answer_prompt`/`system_prompt` says "answer using ONLY the context... if it doesn't contain the answer, say so." That's good for faithfulness, but it doesn't push the model toward *staying focused* on the question. A vague or overly generic answer (or a flat "context doesn't contain the answer") scores badly here, because it doesn't map cleanly back to the specific question asked. Tightening the instruction — e.g. explicitly asking it to directly answer what was asked, without wandering into other retrieved details — is the generation-side fix.

**2. Check retrieval isn't feeding off-topic context** — this is really diagnostic, not a separate fix. Use `context_precision` as the signal: if precision is already high for a question but relevancy is still low, the problem is generation (prompt), not retrieval. If precision is low, the LLM is being handed a mix of relevant and irrelevant chunks, and it's natural for the answer to drift toward whatever's in that noise, pulling it off-topic. In that case, fixing chunking/K (above) often improves `answer_relevancy` for free, without touching the prompt at all — worth re-running `evaluation.py` after a retrieval fix before also tuning the prompt, so you're not solving the same problem twice.

## The general workflow

Change **one** variable (K, chunk size, or prompt), re-run `evaluation.py`, and compare the new row in `results/summary.md` against the old ones. That loop — measure, change one thing, measure again — is how RAG systems actually get better.
