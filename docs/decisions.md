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
