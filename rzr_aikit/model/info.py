from rzr_aikit import model_app
import typer
from typing_extensions import Annotated
from rich.console import Console

from rzr_aikit.utils.spinner import spinner

console = Console()


@model_app.command()
@spinner("Getting information from HuggingFace...", console=console)
def info(
    model_name: Annotated[
        str, typer.Argument(help="organization/model_name")
    ] = "facebook/opt-125m",
    full_config: Annotated[
        bool,
        typer.Option(
            "--full-config", "-f", help="Show full model configuration from config.json"
        ),
    ] = False,
):
    """
    Get detailed information about a model.

    This command fetches and displays comprehensive metadata about a model from
    HuggingFace Hub, including general information (type, size, downloads, license),
    technical specifications (context length, precision, last update), and compatibility
    with local and distributed inference. Optionally displays the full model configuration.

    Examples:

        $ rzr-aikit model info Qwen/Qwen3.5-0.8B
        $ rzr-aikit model info facebook/opt-125m --full-config
    """
    from rzr_aikit.utils.model_classifier import ModelCategory, classify_model
    from rzr_aikit.utils.mlib import get_cuda_total_vram, check_model_fit
    from rzr_aikit.utils.connectivity import check_huggingface_connectivity
    from rzr_aikit import HuggingfaceAccessTokenRequired

    from rich.panel import Panel
    from rich.text import Text
    from humanize import naturalsize

    try:
        has_connection = check_huggingface_connectivity()

        huggingface_info = None
        quant_type = None
        model_config = None
        data_type = None
        weight_size = None
        local = None
        distributed = "-"

        category = classify_model(model_name)

        if category == ModelCategory.DIFFUSION:
            # Standard diffusers pipeline — has model_index.json
            from rzr_aikit.utils.DiffusionModelInfoFetcher import DiffusionModelInfoFetcher
            try:
                fetcher = DiffusionModelInfoFetcher(
                    model_name, allow_internet=has_connection
                )
            except HuggingfaceAccessTokenRequired:
                from rzr_aikit.utils.HuggingfaceHubToken import HuggingfaceHubToken

                hug = HuggingfaceHubToken()
                token = hug.get_access_token()
                fetcher = DiffusionModelInfoFetcher(
                    model_name, access_token=token, allow_internet=has_connection
                )

            if has_connection:
                huggingface_info = fetcher.model_info
            weight_size = fetcher.total_bytes

        elif category in (ModelCategory.BAGEL, ModelCategory.STANDARD):
            # Bagel, standard LLMs, and Mistral-format models (params.json)
            from rzr_aikit.utils.ModelInfoFetcher import ModelInfoFetcher
            try:
                fetcher = ModelInfoFetcher(model_name, allow_internet=has_connection)
            except HuggingfaceAccessTokenRequired:
                from rzr_aikit.utils.HuggingfaceHubToken import HuggingfaceHubToken

                hug = HuggingfaceHubToken()
                token = hug.get_access_token()
                fetcher = ModelInfoFetcher(
                    model_name, access_token=token, allow_internet=has_connection
                )

            if has_connection:
                huggingface_info = fetcher._model_info
            if fetcher.quant_config:
                quant_type = fetcher.quant_config.get("quant_method")
            model_config = fetcher.config
            data_type = fetcher.dtype
            weight_size = fetcher.total_bytes

        local_memory = get_cuda_total_vram()
        dist_memory = get_cuda_total_vram(distributed=True)

        local = check_model_fit(
            local_memory, weight_size, quant_type if quant_type else data_type
        )
        if dist_memory > 0:
            distributed = check_model_fit(
                dist_memory, weight_size, quant_type if quant_type else data_type
            )

        # Basic Info Card with aligned formatting
        basic_info = Text()
        if has_connection:
            basic_info.append(f"{'Type:':<12} ", style="bold")
            basic_info.append(f"{huggingface_info.pipeline_tag}\n")
            basic_info.append(f"{'Downloads:':<12} ", style="bold")
            basic_info.append(f"{huggingface_info.downloads:,}\n")
            basic_info.append(f"{'License:':<12} ", style="bold")
            basic_info.append(
                f"{getattr(huggingface_info.card_data, 'license', 'Unknown')}\n"
            )
        basic_info.append(f"{'Size:':<12} ", style="bold")
        basic_info.append(f"{naturalsize(weight_size, binary=True)}")

        basic_panel = Panel(
            basic_info,
            title="General Info",
            title_align="left",
            border_style="green",
            width=42,
        )

        # Technical Specs Card with aligned formatting
        tech_info = Text()
        if has_connection:
            tech_info.append(f"{'Updated:':<12} ", style="bold")
            tech_info.append(f"{huggingface_info.last_modified}\n", style="dim")
        tech_info.append(f"{'Context Len:':<12} ", style="bold")
        context_len = (
            model_config.get("max_position_embeddings", "Unknown")
            if model_config
            else "Unknown"
        )
        tech_info.append(f"{context_len}\n")
        tech_info.append(f"{'Precision:':<12} ", style="bold")
        precision = (
            quant_type if quant_type else (data_type if data_type else "Unknown")
        )
        tech_info.append(f"{precision}")

        tech_panel = Panel(
            tech_info,
            title="Technical Specs",
            title_align="left",
            border_style="green",
            expand=False,
        )

        # Compatibility Card
        compat_info = Text()
        compat_info.append(f"{'Local:':<12} ", style="bold")
        compat_info.append(f"{local}\n")
        compat_info.append(f"{'Distributed:':<12} ", style="bold")
        compat_info.append(f"{distributed}")

        compat_panel = Panel(
            compat_info,
            title="Compatibility",
            title_align="left",
            border_style="yellow",
            width=42,
        )

        # Display cards vertically (top to bottom)
        console.print(Panel(model_name, border_style="cyan", expand=False))
        console.print(basic_panel)
        console.print(tech_panel)
        console.print(compat_panel)
        if local == "🟡 Limited" or distributed == "🟡 Limited":
            console.print(
                "🟡 The model supports 4-bit on-the-fly quantization using '--quantization bitsandbytes', with reduced accuracy."
            )
        if full_config and model_config:
            # Create a config panel instead of table
            config_text = Text()
            for key, value in model_config.items():
                config_text.append(f"{key}: ", style="bold")
                config_text.append(f"{str(value)}\n", style="dim")

            config_panel = Panel(
                config_text,
                title="Full Configuration",
                title_align="left",
                border_style="blue",
                expand=False,
            )
            console.print(config_panel)

        if dist_memory == 0:
            console.print(
                "[yellow]Not connected to Ray cluster, distributed compatibility skipped.[/]"
            )
        if dist_memory == -1:
            console.print(
                "[yellow]GPU discovery is in progress, distributed compatibility skipped.[/]"
            )
    except Exception as e:
        console.print(f"Incompatible model: {e}")


if __name__ == "__main__":
    info("facebook/opt-125m")
