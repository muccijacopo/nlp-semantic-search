from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import uvicorn
from main import train_model

from query import Query

app = FastAPI(title="Semantic Search App")


@app.get("/query")
def read_root(q: str, topic: str, model: str):
    r = Query.make_query(q, topic=topic, model=model)
    return PlainTextResponse(r)


@app.get("/train")
def read_root(model: str, topic: str):
    train_model(model, topic)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
