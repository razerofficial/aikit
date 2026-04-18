import typer
from typing_extensions import Annotated


def add_help_option():
    def help_option(
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

    return help_option


model_app = typer.Typer(help="Commands to manage models.", add_help_option=False)
cluster_app = typer.Typer(help="Commands to manage clusters.", add_help_option=False)
ui_app = typer.Typer(help="Commands to manage image generation UI.", add_help_option=False)

model_app.callback(invoke_without_command=True)(add_help_option())
cluster_app.callback(invoke_without_command=True)(add_help_option())
ui_app.callback(invoke_without_command=True)(add_help_option())


# exceptions to be used in place of pyone exceptions
class RzrpyError(Exception):
    """Base exception class."""


class RzrpyNoneExistsError(RzrpyError):
    """Raised when a requested object does not exist."""


class RzrpyInternalError(RzrpyError):
    """Raised when an internal error occurs."""


class HuggingfaceAccessTokenRequired(PermissionError):
    """Custom exception for API 401 Unauthorized errors."""


# __version__ = version("rzr")
# https://stackoverflow.com/questions/74656951/how-do-i-embed-the-version-from-pyproject-toml-so-my-package-can-use-it
