import typer

from importing import import_xml_to_csv
from preprocessing import Preprocessing
from query import Query

app = typer.Typer()


@app.command('import')
def start_xml_import():
    import_xml_to_csv()


@app.command('query')
def query(s: str):
    return Query.make_query(s)


@app.command()
def hello():
    print("Hello")


if __name__ == '__main__':
    app()

