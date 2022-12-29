import time
import typer

from importing import import_xml_to_csv
from models import TfIdfModel, LsiModel, LdaModel
from query import Query

app = typer.Typer()


@app.command('import')
def import_command():
    import_xml_to_csv()


@app.command('train')
def train(topic: str = typer.Option(..., help="Topic to train"), model=typer.Option(..., help="Model")):
    start_time = time.time()
    if model == 'tfidf':
        TfIdfModel().train(topic)
    elif model == 'lsi' or model == 'lsa':
        LsiModel().train(topic)
    elif model == 'lda':
        LdaModel().train(topic)
    else:
        print(f"{model} not implemented")
    end_time = time.time()
    print(f"Train execution time: {round(end_time - start_time, 2)}s")


@app.command('query')
def query_command(query: str, topic: str = typer.Option(..., help="Topic search"), model: str = typer.Option(..., help="Model (es. word2vec, tfidf, ...")):
    start_time = time.time()
    Query.make_query(query, topic=topic, model=model)
    end_time = time.time()
    print(f"Query execution time: {round(end_time - start_time, 2)}s")


if __name__ == '__main__':
    app()

