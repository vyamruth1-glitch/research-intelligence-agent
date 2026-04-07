from fastapi import FastAPI
from pydantic import BaseModel
from src.query import query_papers

app = FastAPI()


# Request schema
class QueryRequest(BaseModel):
    question: str
    evaluate: bool = True


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest):
    result = query_papers(
        question=request.question,
        evaluate=request.evaluate
    )
    return result
