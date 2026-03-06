import typer

from src import model_app
from src import cluster_app

from src.model import list
from src.model import run
from src.model import download
from src.model import info
from src.model import remove
from src.model import stop
from src.model import generate
from src.metrics import metrics
from src.ui import ui

from src.cluster import run
from src.cluster import join
from src.cluster import stop
from src.cluster import status

from typing_extensions import Annotated

app = typer.Typer(add_help_option=False, add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    help: Annotated[
        bool,
        typer.Option(
            "--help",
            help="Display help information including usage syntax, available options, and examples.",
        ),
    ] = False,
):
    """A command-line interface for Razer AI compute services."""
    if help or ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


# Register Model commands
app.add_typer(model_app, name="model")

# Register Cluster commands
app.add_typer(cluster_app, name="cluster")

# Register root-level commands
app.command(name="metrics")(metrics.metrics)
app.command(name="ui")(ui)

if __name__ == "__main__":
    app()
