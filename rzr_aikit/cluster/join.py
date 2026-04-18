import typer
from rzr_aikit import cluster_app
from typer.main import TyperCommand


def get_ip_from_ifname(ifname: str) -> str | None:
    import psutil

    addrs = psutil.net_if_addrs()
    if ifname in addrs:
        for snic in addrs[ifname]:
            if snic.family.name == "AF_INET":
                return snic.address
    return None


# Create custom command class that inherits from TyperCommand
class CustomCommand(TyperCommand):
    def format_usage(self, ctx, formatter):
        formatter.write_usage(ctx.command_path, "[OPTIONS] [RAY_START_OPTIONS...]")


@cluster_app.command(
    cls=CustomCommand,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=True,
    short_help="Start a cluster worker on current node to join the cluster",
)
def join(
    ctx: typer.Context,
    ifname: str = typer.Option(
        "eth0",
        "--ifname",
        help="Specify the network interface for NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME",
    ),
    address: str | None = typer.Option(
        None,
        "--address",
        help="Specify Ray head address <Ray head ip>:<Ray head port> to join",
    ),
    cuda_visible_devices: str | None = typer.Option(
        None,
        "--cuda-visible-devices",
        help='Specify the CUDA_VISIBLE_DEVICES string (e.g., "0", "0,1", "1,3")',
    ),
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        help="Specify the Ray backend log level, None or select one from [trace,debug,info,warning,error,fatal]",
    ),
    health_check_timeout_ms: int | None = typer.Option(
        None,
        "--health-check-timeout-ms",
        help="Specify the Ray health check timeout in milliseconds, None will use Ray's internal default 10000 ms",
    ),
):
    """
    Start a Ray cluster worker node to join an existing cluster.

    This command starts a Ray worker node on the current machine and connects it to
    an existing Ray cluster head. You must specify the head node address. Additional
    configuration options include network interface, CUDA GPU visibility, log level,
    and health check timeout. Extra Ray start options can be passed after the
    script-specific options.

    RAY_START_OPTIONS:

    Any additional options will be passed directly to the 'ray start' command.
    These should come after script-specific options or after '--'.
    For example: --num-cpus 4 --num-gpus 2 --block

    Examples:

        $ rzr-aikit cluster join --address 192.168.1.100:6379
        $ rzr-aikit cluster join --ifname enp130s0 --address 192.168.1.100:6379
        $ rzr-aikit cluster join --address 192.168.1.100:6379 --health-check-timeout-ms 60000
        $ rzr-aikit cluster join --address 192.168.1.100:6379 --cuda-visible-devices "0,1"
    """
    options = []

    # --ifname
    if ifname != "eth0":
        options.append("--ifname")
        options.append(ifname)

    # --address
    if address is None:
        print("--address <Ray head ip>:<Ray head port> is required")
        return
    else:
        options.append("--address")
        options.append(address)

    # --cuda-visible-devices
    if cuda_visible_devices:
        options.append("--cuda-visible-devices")
        options.append(cuda_visible_devices)

    # --log-level
    if log_level is not None:
        if log_level not in ["trace", "debug", "info", "warning", "error", "fatal"]:
            print(
                f"--log-level Invalid log level. Must be one of: trace, debug, info, warning, error, fatal"
            )
            return
        else:
            options.append("--log-level")
            options.append(log_level)

    # --health-check-timeout-ms
    if health_check_timeout_ms is not None:
        if not isinstance(health_check_timeout_ms, int) or health_check_timeout_ms <= 0:
            print(f"--health-check-timeout-ms must be a positive integer")
            return
        else:
            options.append("--health-check-timeout-ms")
            options.append(str(health_check_timeout_ms))

    try:
        flags: list[str] = ctx.args

        # check whether --node-ip-address in flags
        if "--node-ip-address" not in flags:
            node_ip_address = get_ip_from_ifname(ifname)
            if node_ip_address is not None:
                options.append("--node-ip-address")
                options.append(node_ip_address)

        cmd = ["start_ray_worker"] + options + flags
        print(f"commands : {cmd}")

        import subprocess

        result = subprocess.run(cmd, check=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return
    except FileNotFoundError:
        print("Command not found: start_ray_head")
        return
