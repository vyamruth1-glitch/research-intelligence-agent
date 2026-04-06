from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

def rewrite_query(original_question: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""You are helping improve a search query for retrieving
relevant chunks from AI/ML research papers.

Rewrite the following question to be more specific and retrieval-friendly.
- Make implicit concepts explicit
- Break compound questions into the core information need
- Keep it concise — one focused sentence
- Preserve the original meaning exactly
- Do NOT expand, redefine, or reinterpret technical acronyms (e.g. RAG, LLM) unless they are already expanded in the original question
- Do NOT answer the question, only rewrite it

Original question: {original_question}

Rewritten query:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    rewritten = response.choices[0].message.content.strip()
    return rewritten


if __name__ == "__main__":
    test = "How do different papers approach reducing hallucination in RAG?"
    rewritten = rewrite_query(test)
    print(f"Original:  {test}")
    print(f"Rewritten: {rewritten}")
