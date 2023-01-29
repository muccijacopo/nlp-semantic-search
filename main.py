import time
import typer

from importing import import_xml_to_csv
from models import TfIdfModel, LsiModel, LdaModel, LsiTfidfModel, Doc2Vec, Word2VecModel, DistilBertModel
from query import Query


def train_model(model: str, topic: str):
    start_time = time.time()
    if model == 'word2vec':
        Word2VecModel().train(topic)
    elif model == 'tfidf':
        TfIdfModel().train(topic)
    elif model == 'lsi':
        LsiModel().train(topic)
    elif model == 'lsi-tfidf':
        LsiTfidfModel().train(topic)
    elif model == 'lda':
        LdaModel().train(topic)
    elif model == 'doc2vec':
        Doc2Vec().train(topic)
    elif model == 'distilbert':
        DistilBertModel().train(topic)
    else:
        print(f"{model} not implemented")
    end_time = time.time()
    print(f"Train execution time: {round(end_time - start_time, 2)}s")


app = typer.Typer()


@app.command('import')
def import_command():
    import_xml_to_csv()


@app.command('train')
def train(topic: str = typer.Option(..., help="Topic to train"), model=typer.Option(..., help="Model")):
    train_model(model, topic)


@app.command('query')
def query_command(query: str, topic: str = typer.Option(..., help="Topic search"), model: str = typer.Option(..., help="Model (es. word2vec, tfidf, ...")):
    start_time = time.time()
    Query.make_query(query, topic=topic, model=model, generate_text=True)
    end_time = time.time()
    print(f"Query execution time: {round(end_time - start_time, 2)}s")


if __name__ == '__main__':
    app()

