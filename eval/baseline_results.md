## Baseline Evaluation — Naive RAG

### Q1: How do different papers approach reducing hallucination in RAG?
Answer: According to the provided context, there are two approaches mentioned for reducing hallucination in Retrieval-Augmented Generation (RAG) pipelines: filtering context and incorporating structured knowledge graphs (GraphRAG). Filtering context helps remove irrelevant or poorly chosen context, while GraphRAG uses finer-grained knowledge representation to reduce interference from unrelated information.

Sources cited by system: hallucination_1.pdf, hallucination_2.pdf, chunking_2.pdf
Top relevance score: 0.771

Honest assessment:
- Answered the question: Partially
- Evidence of hallucination: No obvious hallucination, but answer is shallow
- Cross-paper comparison attempted: Weakly
- What was weak: The answer did not clearly compare multiple papers. It flattened the retrieved evidence into a short synthesis instead of distinguishing which paper proposed what. Retrieval seems decent, but cross-paper reasoning is weak.
### Q2: How do different papers approach chunk size selection in RAG systems?
Answer: The retrieved context does not contain enough information to answer this.

Sources cited by system: chunking_1.pdf
Top relevance score: 0.869

Honest assessment:
- Answered the question: No
- Evidence of hallucination: No
- Cross-paper comparison attempted: No
- What was weak: Retrieval found chunking-related material with a high relevance score, but the system still could not answer the comparative question. This suggests that retrieval relevance alone does not guarantee enough evidence for synthesis across papers.
### Q3: How do reranking methods improve retrieval quality compared to basic similarity search?
Answer: The system explained that reranking improves retrieval by resolving fine-grained distinctions among topically similar candidates, improving ranking quality and correcting encoder mistakes. It also cited performance gains such as higher Hit@1 and better compact pipeline performance.

Sources cited by system: reranking_2.pdf
Top relevance score: 0.837

Honest assessment:
- Answered the question: Partially to mostly
- Evidence of hallucination: No obvious hallucination, but answer may over-generalize from a single paper
- Cross-paper comparison attempted: No
- What was weak: The answer sounded strong and specific, but it appears grounded in only one paper. It did not compare multiple reranking methods or clearly separate general principles from one-paper findings.
### Q4: How do recent papers evaluate faithfulness in RAG or LLM-based systems?
Answer: The system explained that recent papers evaluate faithfulness using classifiers over chain-of-thought reasoning traces, different judging pipelines, and related analyses such as sycophancy and reasoning behavior. It also noted that evaluation results can vary significantly depending on the evaluation method used.

Sources cited by system: evaluation_2.pdf, hallucination_2.pdf, rag_survey_1.pdf
Top relevance score: 0.822

Honest assessment:
- Answered the question: Partially to mostly
- Evidence of hallucination: No obvious hallucination, but answer may blur adjacent concepts
- Cross-paper comparison attempted: Somewhat
- What was weak: The answer retrieved relevant evaluation-related material, but it mixed nearby ideas like chain-of-thought faithfulness, sycophancy, and broader LLM behavior without tightly structuring the answer around the exact question scope.
### Q5: What are the main trade-offs between retrieval quality and response generation quality in RAG pipelines?
Answer: The system described trade-offs such as retrieval granularity vs. generation context construction, information fragmentation vs. long-range dependencies, and retrieval noise leading to irrelevant context injection. These trade-offs were primarily grounded in chunking-related analysis.

Sources cited by system: chunking_1.pdf
Top relevance score: 0.815

Honest assessment:
- Answered the question: Partially
- Evidence of hallucination: No obvious hallucination
- Cross-paper comparison attempted: No
- What was weak: The answer focused on trade-offs from a single paper and presented them as general RAG trade-offs. It did not synthesize multiple perspectives or capture the broader conceptual trade-offs across papers.

---
---
---
---
---
