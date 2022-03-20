"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -muncertainty_motion_prediction` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``uncertainty_motion_prediction.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``uncertainty_motion_prediction.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import typer


app = typer.Typer()


@app.command()
def entry(
    name: str,
    age: int = typer.Option(2, "-a", "--age", help="Age."),
):
    """Default entrypoint."""
    typer.echo(f"Hello, I'm {name}. I'm {age} years old.")


def main():
    app()


if __name__ == "__main__":
    main()
