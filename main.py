import time
from typing import Optional

import typer

from importing import import_xml_to_csv
from query import Query

app = typer.Typer()


@app.command('import')
def import_command():
    import_xml_to_csv()


@app.command('query')
def query_command(query: str, topic: str = typer.Option(..., help="Topic search"), model: str = typer.Option(..., help="Model (es. word2vec, tfidf, ...")):
    start_time = time.time()
    r = Query.make_query(query, topic=topic, model=model)
    print(r)
    print(f"Query execution time: {round(time.time() - start_time, 2)}s")


@app.command()
def hello_command():
    print("Hello")


if __name__ == '__main__':
    app()

