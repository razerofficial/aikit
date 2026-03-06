from src import model_app
import typer
from typing_extensions import Annotated
from rich.console import Console

console = Console()


@model_app.command()
def generate(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 1.0,
    model: Annotated[str, typer.Option(help="Model to use for generation")] = "",
):
    """
    Generate a completion using a model.

    This command sends a prompt to the running vLLM server and streams the generated
    response to the console. It displays token generation statistics including
    prompt tokens, completion tokens, total time, and throughput. If no model is
    specified, it uses the first available running model.

    Examples:

        $ rzr-aikit model generate "Tell me about AI in Razer"
        $ rzr-aikit model generate "Explain quantum computing" --max-tokens 512
        $ rzr-aikit model generate "Write a poem" --temperature 0.9 --top-p 0.95
        $ rzr-aikit model generate "Hello" --model facebook/opt-125m
    """
    try:
        import time

        with console.status("Generating output..."):
            from util.mlib import get_running_models, comp_generate, image_generate

            models = [model.id for model in get_running_models()]

            if model:
                if model not in models:
                    console.print(f"Model '{model}' is either invalid or not running.")
                    return
                models = [model]

            if len(models) == 0:
                console.print(
                    "No models are currently running. Please start a model first."
                )
                return

        for model_id in models:
            from rich.panel import Panel
            from rich.table import Table

            # Show model name panel
            console.print(
                Panel(
                    model_id,
                    border_style="green",
                    title_align="left",
                    expand=False,
                )
            )

            start = time.time()
            try:
                stream = comp_generate(
                    model=model_id,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                    stream_options={"include_usage": True},
                )
            except Exception as e:
                import base64
                import os

                with console.status("Generating image..."):
                    img = image_generate(
                        model=model_id,
                        prompt=prompt,
                        n=1,
                        size="512x512",
                    )

                image_bytes = base64.b64decode(img.data[0].b64_json)
                with open("output.png", "wb") as f:
                    f.write(image_bytes)

                timespan = time.time() - start

                if "JPY_PARENT_PID" not in os.environ:
                    os.system("chafa --symbols=block --colors=full output.png")

                stats_table = Table(show_header=True, header_style="bold", box=None)
                stats_table.add_column("Filename")
                stats_table.add_column("Time")
                stats_table.add_row("", "")  # blank row for spacing
                stats_table.add_row("output.png", f"{timespan:.2f}s")

                console.print(
                    Panel(
                        stats_table,
                        border_style="yellow",
                        expand=False,
                    )
                )
                console.print("[yellow]Open image file to view in full quality...[/]")
                return

            # Stream completion text directly to console
            stats = None
            for token in stream:
                if token.usage:
                    stats = token.usage
                else:
                    console.print(token.choices[0].text, end="", highlight=False)

            console.print()

            # Show final statistics panel
            timespan = time.time() - start

            if stats:
                stats_table = Table(show_header=True, header_style="bold", box=None)
                stats_table.add_column("Prompt tokens")
                stats_table.add_column("Completion tokens")
                stats_table.add_column("Total")
                stats_table.add_column("Time")
                stats_table.add_column("Throughput")
                stats_table.add_row("", "", "", "", "")  # blank row for spacing

                stats_table.add_row(
                    str(stats.prompt_tokens),
                    str(stats.completion_tokens),
                    str(stats.total_tokens),
                    f"{timespan:.2f}s",
                    f"{stats.total_tokens / timespan:.2f} tok/s",
                )

                console.print(
                    Panel(
                        stats_table,
                        title="[yellow]Statistics[/]",
                        title_align="left",
                        border_style="yellow",
                        expand=False,
                    )
                )
    except Exception as e:
        console.print(f"Error during generation: {e}")
