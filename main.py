import time
import typer

from importing import import_xml_to_csv
from query import Query

app = typer.Typer()


@app.command('import')
def start_xml_import():
    import_xml_to_csv()


@app.command('query')
def query(s: str):
    start_time = time.time()
    r = Query.make_query(s)
    print(r)
    print(f"Query execution time: {round(time.time() - start_time, 2)}s")


@app.command()
def hello():
    print("Hello")


if __name__ == '__main__':
    app()

