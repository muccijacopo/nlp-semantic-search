from typing import Union

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import uvicorn

from query import Query

app = FastAPI(title="Semantic Search App")


@app.get("/query")
def read_root(q: str, topic: str, model: str):
    r = Query.make_query(q, topic=topic, model=model)
    return PlainTextResponse(r)


@app.get("/train")
def read_root():
    return "Not yet"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
