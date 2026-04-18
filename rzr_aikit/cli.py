import typer

from rzr_aikit import model_app
from rzr_aikit import cluster_app
from rzr_aikit import ui_app

from rzr_aikit.model import list
from rzr_aikit.model import run
from rzr_aikit.model import download
from rzr_aikit.model import info
from rzr_aikit.model import remove
from rzr_aikit.model import stop
from rzr_aikit.model import generate
from rzr_aikit.metrics import metrics
from rzr_aikit.ui import run
from rzr_aikit.ui import stop

from rzr_aikit.cluster import run
from rzr_aikit.cluster import join
from rzr_aikit.cluster import stop
from rzr_aikit.cluster import status

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

# Register UI commands
app.add_typer(ui_app, name="ui")

# Register root-level commands
app.command(name="metrics")(metrics.metrics)

if __name__ == "__main__":
    app()
