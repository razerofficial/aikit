import typer
from src import cluster_app
from typer.main import TyperCommand
from rich.console import Console
from rich.text import Text

from util.spinner import spinner

console = Console()


# Create custom command class that inherits from TyperCommand
class CustomCommand(TyperCommand):
    def format_usage(self, ctx, formatter):
        formatter.write_usage(ctx.command_path, "[OPTIONS] [RAY_STOP_OPTIONS...]")


@cluster_app.command(
    cls=CustomCommand,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=True,
    short_help="Stop cluster processes on current node",
)
@spinner("Stopping Ray processes...", console=console)
def stop(
    ctx: typer.Context,
    force: bool = typer.Option(
        False, "--force", is_flag=True, help="Ray will send SIGKILL instead of SIGTERM"
    ),
    grace_period: int | None = typer.Option(
        None,
        "--grace-period",
        help="The time in seconds ray waits for processes to be properly terminated",
    ),
):
    """
    Stop Ray cluster processes on the current node.

    This command terminates all Ray processes running on the current machine,
    whether it's a head node or a worker node. You can optionally force immediate
    termination with SIGKILL or specify a grace period for graceful shutdown.
    Additional Ray stop options can be passed after the script-specific options.

    RAY_STOP_OPTIONS:

    Any additional options will be passed directly to the 'ray stop' command.
    For example: --log-style pretty --log-color auto --verbose

    Examples:

        $ rzr-aikit cluster stop
        $ rzr-aikit cluster stop --force
        $ rzr-aikit cluster stop --grace-period 30
    """
    options = []
    flags: list[str] = ctx.args

    # --force
    if force:
        if "-f" not in flags:
            options.append("--force")

    # --grace-period
    if grace_period is not None:
        if "-g" not in flags:
            if not isinstance(grace_period, int) or grace_period <= 0:
                print(f"--grace-period must be a positive integer")
                return
            else:
                options.append("--grace-period")
                options.append(str(grace_period))

    try:
        cmd = ["ray", "stop"] + options + flags

        import subprocess

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        if result.stdout:
            console.print(Text.from_ansi(result.stdout))

        return result
    except subprocess.CalledProcessError as e:
        console.print(f"Command failed with exit code {e.returncode}")
        return
    except FileNotFoundError:
        console.print("Command not found: start_ray_head")
        return
