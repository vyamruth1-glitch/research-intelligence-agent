from typing import Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator
from src.query import query_papers

app = FastAPI()

VALID_TOPICS = Literal["RAG", "LLM_eval", "agents", "embeddings", "hallucination", "RAG_eval"]


class QueryRequest(BaseModel):
    question: str
    evaluate: bool = True
    # Optional payload filters — applied at the Qdrant retrieval layer before
    # semantic search, so they restrict the candidate space rather than post-filtering.
    # topic must be one of the six ingested topic labels; years are 1990–2030.
    topic: Optional[VALID_TOPICS] = None
    year_from: Optional[int] = Field(default=None, ge=1990, le=2030)
    year_to: Optional[int] = Field(default=None, ge=1990, le=2030)

    @model_validator(mode="after")
    def year_range_consistent(self):
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError("year_from must be <= year_to")
        return self


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest):
    result = query_papers(
        question=request.question,
        evaluate=request.evaluate,
        topic=request.topic,
        year_from=request.year_from,
        year_to=request.year_to,
    )
    return result
