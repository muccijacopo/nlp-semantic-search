import time
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from main import train_model
from query import Query

app = FastAPI(title="Semantic Search API")
app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.get("/query")
def read_root(q: str, topic: str, model: str, format: str):
    start_time = time.time()
    r = Query.make_query(q, topic=topic, model=model, generate_text=format == 'answer')
    end_time = time.time()
    return JSONResponse({
        'documents': r,
        'response_time': end_time - start_time
    })


@app.get("/train")
def read_root(model: str, topic: str):
    train_model(model, topic)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
