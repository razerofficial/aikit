import typer
from src import cluster_app
from rich.console import Console

from util.spinner import spinner

console = Console()


@cluster_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=True,
    short_help="Print cluster status",
)
@spinner("Fetching cluster status...", console=console)
def status(
    ctx: typer.Context,
    address: str | None = typer.Option(
        None, "--address", help="Override the address to connect to"
    ),
):
    """
    Display the current status of the Ray cluster.

    This command prints detailed information about the Ray cluster, including node
    status, resource utilization, and active workers. If multiple Ray clusters are
    running or you need to check a remote cluster, specify the head node address
    using the --address option.

    Examples:

        $ rzr-aikit cluster status
        $ rzr-aikit cluster status --address 192.168.1.100:6379
    """
    options = []

    # --address
    if address:
        options.append("--address")
        options.append(address)

    try:
        cmd = ["ray", "status"] + options

        import subprocess

        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode == 1:
            # Regex to detect `raise ConnectionError`
            import re

            if re.search(r"raise\s+ConnectionError", proc.stderr):
                console.print(
                    "Failed to connect: either multiple or no Ray clusters found.\n"
                )
                console.print(
                    "If a Ray cluster is running, please specify the address of the Ray head node using --address:"
                )
                console.print(
                    "rzr-aikit cluster status --address <Ray head ip>:<Ray head port>",
                    highlight=False,
                    style="yellow",
                )
        else:
            console.print(proc.stdout)
    except subprocess.CalledProcessError as e:
        console.print(f"Command failed with exit code {e.returncode}")
        return
    except FileNotFoundError:
        console.print("Command not found: ray status")
        return
