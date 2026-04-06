
## Environment Decision
Upgraded to Python 3.11 from 3.9.
Why: LlamaIndex uses union type syntax (X | Y) introduced in 3.10+.
Pinning old packages would create compounding compatibility issues.

## Embedding Model Decision
Using BAAI/bge-small-en-v1.5 via HuggingFace locally.
Why: No OpenAI dependency, free to run, strong performance 
on technical/scientific text, and we control the model explicitly 
rather than relying on framework defaults.

## Chunking Observation — Week 2
174 document sections → 458 chunks at chunk_size=512, overlap=50.
Average ~2.6 chunks per document section.
Hypothesis: some sections are being over-chunked, 
possibly splitting mid-argument in dense paper sections.
Will revisit chunk size in Week 3 when we measure retrieval quality.

## Temperature Decision
Using temperature=0.1 for query answering.
Why: lower temperature reduces creativity and helps the model stay
closer to retrieved context, improving faithfulness for research QA.
## LLM Model Decision
Using llama-3.1-8b-instant via Groq.
Why: llama3-8b-8192 was decommissioned. 
8b-instant gives fast inference, free tier, 
sufficient reasoning for retrieval-grounded answers.
Would evaluate llama-3.1-70b for better cross-paper 
reasoning in later weeks if quality is insufficient.
## Query Rewriting Observation — Week 3

Original: How do different papers approach reducing hallucination in RAG?

Rewritten: What are the specific methods and techniques employed by recent AI/ML research papers to mitigate or reduce hallucination in Retrieval-Augmented Generation (RAG) models?

Observation:
The rewritten query is more explicit and retrieval-friendly, clearly specifying methods, techniques, and domain context.

However, it slightly loses the comparative intent ("different papers"), shifting toward a general summary question.

Insight:
Query rewriting improves clarity but can unintentionally alter the question’s original intent. This needs to be monitored, especially for comparison-type queries.
Original: What are the main trade-offs between retrieval quality and response generation quality in RAG pipelines?

Rewritten: What are the key trade-offs between retrieval performance and response generation capabilities in Retrieval-Augmented Generation (RAG) models?

Observation:
This rewrite preserved the original intent more successfully than the first example while making the wording slightly cleaner and more retrieval-friendly.

However, there is still a subtle wording shift from "quality" to "capabilities," which may slightly broaden the meaning.

Insight:
The query rewriter appears helpful overall, but its outputs should be checked for small intent shifts before trusting them automatically.
## Improvement 1 Test — Query Rewriting in Full Pipeline

Test question:
How do different papers approach reducing hallucination in RAG?

Rewritten query:
What are the specific methods and techniques employed by recent AI/ML research papers to mitigate or reduce hallucination in Retrieval-Augmented Generation (RAG) models?

Observed result:
Retrieval relevance improved compared to the baseline (top relevance score increased), and the answer included slightly broader supporting material.

However, the core failure mode remained:
the system still blended multiple retrieved ideas into one smooth synthesis rather than clearly distinguishing how different papers approached the problem.

Conclusion:
Query rewriting improved retrieval precision, but did not solve cross-paper synthesis faithfulness or source nuance preservation by itself.
## Improvement 2 Test — Source Diversity Retrieval

Test question:
How do different papers approach reducing hallucination in RAG?

Observed result:
Source diversity improved significantly. The system retrieved chunks from more distinct papers instead of leaning heavily on one or two highly similar sources.

This improved cross-paper coverage and reduced the tendency to over-generalize from a single paper.

Trade-off observed:
Some of the additional papers were only loosely relevant, leading the answer to include several "does not explicitly discuss this" style statements.

Conclusion:
Enforcing source diversity improved coverage and reduced source domination, but also introduced weaker evidence into the answer. This suggests diversity retrieval is useful, but needs stronger prompting to help the model separate strong evidence from weak or adjacent evidence.
## Query Rewriting Failure Case — Acronym Drift

Observed failure:
The query rewriter incorrectly expanded "RAG" as
"Reinforcement Learning from Alternative Representations"
instead of "Retrieval-Augmented Generation".

Impact:
This changed the meaning of the user’s question, degraded retrieval quality, and caused the system to retrieve weaker or less relevant evidence.

Insight:
LLM-based query rewriting can improve retrieval precision, but it can also introduce semantic drift by rewriting technical acronyms incorrectly.

Engineering takeaway:
Query rewriting is useful, but should not be trusted blindly. Technical terms and acronyms need guardrails or preservation rules to avoid meaning corruption.
