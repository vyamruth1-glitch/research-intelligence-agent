# Week 3 Evaluation — Improved RAG System

## Q1: How do different papers approach reducing hallucination in RAG?

Answer: Based on the provided sources, different papers approach reducing hallucination in RAG in several ways.

- `chunking_2.pdf` suggests that Retrieval Augmentation (RAG) reduces hallucination by allowing the model to draw on up-to-date or domain-specific knowledge instead of relying only on parametric memory.
- `chunking_2.pdf` also mentions GraphRAG, where structured knowledge graphs help the model reason over relevant facts more precisely.
- `chunking_1.pdf` proposes Marker-RAG (M-RAG), a chunk-free retrieval strategy, but does not directly focus on hallucination reduction.
- `chunking_1.pdf` also mentions Self-RAG, though it does not explicitly explain how it reduces hallucination.
- `hallucination_1.pdf` mentions knowledge graphs and related techniques, but is only indirectly relevant to the question.

Overall, the improved system shows that some papers directly discuss hallucination reduction strategies, while others are only adjacent or partially relevant.

Sources cited by system: ['hallucination_2.pdf', 'hallucination_1.pdf', 'chunking_1.pdf', 'chunking_2.pdf']

Top relevance score: 0.844

Honest assessment:
- Answered the question: Partially
- Evidence of hallucination: N
- Cross-paper comparison attempted: Y
- What was weak: The answer separated papers much better than baseline, but still did not strongly compare papers against each other. Some weakly related sources were still included due to diversity retrieval.

Improvement vs baseline:
The improved system clearly separated claims by source instead of blending them into one smooth answer. It also became more honest about when papers were only indirectly relevant, which improved faithfulness. Compared to baseline, attribution and source-awareness improved significantly, although direct comparison across papers is still limited.


---

## Q2: How do different papers approach chunk size selection in RAG systems?

Answer: Retrieved context is insufficient — this question needs broader coverage.

Sources cited by system: ['chunking_1.pdf']

Top relevance score: 0.86

Honest assessment:
- Answered the question: N
- Evidence of hallucination: N
- Cross-paper comparison attempted: N
- What was weak: The system retrieved chunking-related content, but not enough direct evidence about how multiple papers approach chunk size selection specifically. Retrieval relevance was high, but answerability remained low.

Improvement vs baseline:
The improved system became more honest about insufficiency instead of trying to force a weak or speculative answer. This is a faithfulness improvement, even though the question is still not answerable with the current paper set.

---

## Q3: How do reranking methods improve retrieval quality compared to basic similarity search?

Answer: The system found that reranking improves retrieval quality by resolving fine-grained distinctions among topically similar candidates and correcting ranking errors made by the initial encoder stage. It identified evidence showing reranking can improve top-ranked retrieval quality and recover relevant candidates missed by basic similarity-based ordering. However, the answer was still mostly grounded in one paper rather than a true cross-paper comparison.

Sources cited by system: ['reranking_2.pdf']

Top relevance score: 0.84

Honest assessment:
- Answered the question: Partially
- Evidence of hallucination: N
- Cross-paper comparison attempted: N
- What was weak: The answer was grounded and relevant, but still relied mostly on one source. It explained reranking well, but did not compare different papers or methods broadly.

Improvement vs baseline:
The improved system stayed grounded and avoided overclaiming, but this question remained largely single-source. Compared to baseline, the answer quality is similar in correctness, but slightly better in caution and source attribution.

---

## Q4: How do recent papers evaluate faithfulness in RAG or LLM-based systems?

Answer: The system found that recent papers evaluate faithfulness using classifier-based approaches applied to reasoning traces, along with analysis of chain-of-thought behavior and model alignment. It identified that different classifiers can produce different faithfulness scores, showing that faithfulness is not always a clean or objective measurement. The answer included multiple sources, but still leaned more toward LLM reasoning faithfulness than RAG-specific evaluation.

Sources cited by system: ['evaluation_2.pdf', 'hallucination_2.pdf', 'rag_survey_1.pdf']

Top relevance score: 0.82

Honest assessment:
- Answered the question: Partially
- Evidence of hallucination: N
- Cross-paper comparison attempted: Y
- What was weak: The answer was relevant and source-grounded, but scope drift still occurred. It partially shifted from RAG faithfulness into broader LLM faithfulness and reasoning trace evaluation.

Improvement vs baseline:
The improved system handled the question more carefully and attributed evidence better, but still showed some scope drift. Compared to baseline, faithfulness improved, though retrieval still pulled adjacent concepts rather than staying tightly focused on RAG-specific evaluation.

---

## Q5: What are the main trade-offs between retrieval quality and response generation quality in RAG pipelines?

Answer: The system identified several trade-offs between retrieval quality and generation quality in RAG systems, including retrieval granularity versus generation context construction, information fragmentation versus long-range coherence, and retrieval noise versus useful contextual support. It linked these trade-offs to chunking strategy and retrieval design choices. However, the answer was still based mostly on one paper and did not strongly compare different papers' perspectives.

Sources cited by system: ['chunking_1.pdf']

Top relevance score: 0.82

Honest assessment:
- Answered the question: Partially
- Evidence of hallucination: N
- Cross-paper comparison attempted: N
- What was weak: The answer was relevant and grounded, but still mostly single-source. It captured the trade-off concept well, but lacked broader multi-paper synthesis.

Improvement vs baseline:
The improved system remained grounded and cautious, but this question still behaved like a strong single-paper retrieval task rather than a true multi-paper synthesis task. Compared to baseline, the answer is slightly more faithful and better framed, but not dramatically different.
