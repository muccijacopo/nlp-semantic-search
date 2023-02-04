import json

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from main import train_model

from query import Query

app = FastAPI(title="Semantic Search API")
app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.get("/query")
def read_root(q: str, topic: str, model: str, format: str):
    r = Query.make_query(q, topic=topic, model=model, generate_text=format == 'answer')
    return JSONResponse(r)


@app.get("/train")
def read_root(model: str, topic: str):
    train_model(model, topic)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
