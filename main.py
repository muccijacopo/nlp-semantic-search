import os
import typer

from importing import import_xml_to_csv

app = typer.Typer()


@app.command('import')
def start_xml_import():
    import_xml_to_csv()


@app.command()
def hello():
    print("Hello")


if __name__ == '__main__':
    app()

