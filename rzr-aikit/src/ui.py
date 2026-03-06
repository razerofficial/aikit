import typer
from typing_extensions import Annotated
from rich.console import Console

import base64
import requests
from io import BytesIO
from PIL import Image

console = Console()


def ui(
    server: Annotated[
        str,
        typer.Option(help="Server URL"),
    ] = "http://localhost:8000",
    port: Annotated[
        int,
        typer.Option(help="Gradio port"),
    ] = 7860,
):
    """
    Launch the UI for image generation.

    This command starts a local Gradio web UI for image generation and connects
    it to a running server endpoint. You can configure the backend server URL
    and the Gradio port through command-line arguments.

    Options:

        --server  Server URL to use for image generation requests
                  (default: http://localhost:8000)
        --port    Local port for the Gradio app
                  (default: 7860)

    Examples:

        $ rzr-aikit ui
        $ rzr-aikit ui --server http://127.0.0.1:8000
        $ rzr-aikit ui --port 7861
    """
    try:
        with console.status(f"Connecting to server: {server}"):
            gradio_app = create_gradio_app(server)
            gradio_app.launch(server_port=port)
            return
    except Exception as e:
        console.print(f"Error launching gradio server: {e}")
        return


def generate_image(
    prompt: str,
    height: int,
    width: int,
    steps: int,
    cfg_scale: float,
    seed: int | None,
    negative_prompt: str,
    server_url: str,
    num_outputs_per_prompt: int = 1,
) -> Image.Image | None:
    """Generate an image using the chat completions API."""
    import gradio as gr

    messages = [{"role": "user", "content": prompt}]

    # Build extra_body with generation parameters
    extra_body = {
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "true_cfg_scale": cfg_scale,
    }
    if seed is not None and seed >= 0:
        extra_body["seed"] = seed
    if negative_prompt:
        extra_body["negative_prompt"] = negative_prompt
    # Keep consistent with run_curl_text_to_image.sh, always send num_outputs_per_prompt
    extra_body["num_outputs_per_prompt"] = num_outputs_per_prompt

    # Build request payload
    payload = {"messages": messages, "extra_body": extra_body}

    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list) and len(content) > 0:
            image_url = content[0].get("image_url", {}).get("url", "")
            if image_url.startswith("data:image"):
                _, b64_data = image_url.split(",", 1)
                image_bytes = base64.b64decode(b64_data)
                return Image.open(BytesIO(image_bytes))

        return None

    except Exception as e:
        print(f"Error: {e}")
        raise gr.Error(f"Generation failed: {e}")


def create_gradio_app(server_url: str):
    """
    Create the Gradio image generation interface.

    The UI includes prompt controls, generation settings, curated examples,
    and an output image panel. Generated requests are sent to the provided
    server URL.

    Args:
        server_url: Backend server URL used by the image generation call.

    Returns:
        A configured Gradio Blocks app instance.
    """
    import gradio as gr
    from util.mlib import get_running_models

    Razer = gr.themes.Base(
        primary_hue=gr.themes.Color(
            c100="#dcfce7",
            c200="#b4efab",
            c300="#86efac",
            c400="#4ade80",
            c50="#f0fdf4",
            c500="#44d62c",
            c600="#44d62c",
            c700="#73e161",
            c800="#1b5612",
            c900="#0e2b09",
            c950="#071604",
        ),
        neutral_hue=gr.themes.Color(
            c100="#eeeeee",
            c200="#cccccc",
            c300="#bbbbbb",
            c400="#999999",
            c50="#ffffff",
            c500="#888888",
            c600="#555555",
            c700="#333333",
            c800="#222222",
            c900="#111111",
            c950="#050505",
        ),
        font=[
            gr.themes.GoogleFont("Geist"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ],
        font_mono=[
            gr.themes.GoogleFont("Geist"),
            "ui-monospace",
            "Consolas",
            "monospace",
        ],
        spacing_size=gr.themes.Size(
            lg="16px", md="12px", sm="8px", xl="24px", xs="4px", xxl="40px", xxs="2px"
        ),
        radius_size=gr.themes.Size(
            lg="12px", md="8px", sm="4px", xl="16px", xs="2px", xxl="24px", xxs="1px"
        ),
    ).set(
        button_large_radius="*radius_md",
        button_primary_text_color_dark="black",
        button_large_text_size="20px",
        button_large_text_weight="500",
        input_placeholder_color_dark="*neutral_400",
        block_padding="10px",
        input_padding="*spacing_lg",
        layout_gap="*spacing_lg",
        block_radius="*radius_md",
        input_radius="*radius_lg",
    )

    def get_current_model_html():
        """Get current running model name."""
        models = [model.id for model in get_running_models()]
        current_model = models[0] if len(models) > 0 else "No model running"
        return f'<h3 style="font-weight: 400;">Current model: {current_model}</h3>'

    with gr.Blocks(
        title="Razer Image Generation",
        theme=Razer,
        css="div.svelte-1vd8eap:has(#prompt-textbox) { border: 0 !important; border-radius: 24px !important; }",
    ) as gradio_app:
        # Static header content
        gr.HTML("""
                <div align="left">
                <img src="https://iprototypes.com/grafana/pinkmanlogo.svg" width="900">
                </div>
                <h1 style="margin-top: 5px; font-weight: 500;">Generate images with local running models</h1>
                """)
        model_display = gr.HTML()
        
        gradio_app.load(fn=get_current_model_html, outputs=model_display)

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    show_label=False,
                    placeholder="Describe the image you want to generate...",
                    lines=7,
                    elem_id="prompt-textbox",
                )

                with gr.Accordion("Additional Settings", open=False):
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Describe what you don't want...",
                        lines=1,
                    )
                    with gr.Group():
                        with gr.Row():
                            gr.Textbox(
                                value="Height",
                                show_label=False,
                                interactive=False,
                                container=False,
                            )
                            height = gr.Number(
                                label="Height",
                                minimum=256,
                                maximum=2048,
                                value=512,
                                step=64,
                                show_label=False,
                                container=False,
                            )
                    with gr.Group():
                        with gr.Row():
                            gr.Textbox(
                                value="Width",
                                show_label=False,
                                interactive=False,
                                container=False,
                            )
                            width = gr.Number(
                                label="Width",
                                minimum=256,
                                maximum=2048,
                                value=512,
                                step=64,
                                show_label=False,
                                container=False,
                            )
                    with gr.Group():
                        with gr.Row():
                            gr.Textbox(
                                value="Inference Steps",
                                show_label=False,
                                interactive=False,
                                container=False,
                            )
                            steps = gr.Number(
                                label="Inference Steps",
                                minimum=5,
                                maximum=100,
                                value=10,
                                step=5,
                                show_label=False,
                                container=False,
                            )
                    with gr.Group():
                        with gr.Row():
                            gr.Textbox(
                                value="CFG Scale",
                                show_label=False,
                                interactive=False,
                                container=False,
                            )
                            cfg_scale = gr.Number(
                                label="True CFG Scale",
                                minimum=0.0,
                                maximum=20.0,
                                value=4.0,
                                step=0.5,
                                show_label=False,
                                container=False,
                            )
                    with gr.Group():
                        with gr.Row():
                            gr.Textbox(
                                value="Random Seed",
                                show_label=False,
                                interactive=False,
                                container=False,
                            )
                            seed = gr.Number(
                                label="Random Seed (-1 for random)",
                                value=-1,
                                precision=0,
                                show_label=False,
                                container=False,
                            )

                generate_btn = gr.Button("Generate Image", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Generated Image",
                    type="pil",
                    height=512,
                )

        # Examples
        gr.Examples(
            examples=[
                [
                    "Fluffy cat sitting on glowing Razer keyboard wearing Kraken kitty headphones",
                    "",
                    512,
                    512,
                    9,
                    0.0,
                    -1,
                ],
                [
                    "Cute sea otter wearing a Faker esports jersey holding a basketball",
                    "",
                    512,
                    512,
                    9,
                    0.0,
                    -1,
                ],
                [
                    "Cyberpunk style futuristic city with neon lights and Razer HQ building",
                    "blurry, low quality",
                    512,
                    512,
                    9,
                    0.0,
                    -1,
                ],
                [
                    "Chinese ink painting of a stylized Razer three-headed snake logo",
                    "",
                    512,
                    512,
                    9,
                    0.0,
                    -1,
                ],
            ],
            inputs=[prompt, negative_prompt, height, width, steps, cfg_scale, seed],
        )

        generate_btn.click(
            fn=lambda p, h, w, st, c, se, n: generate_image(
                p,
                h,
                w,
                st,
                c,
                se if se >= 0 else None,
                n,
                server_url,
                1,
            ),
            inputs=[prompt, height, width, steps, cfg_scale, seed, negative_prompt],
            outputs=[output_image],
        )

    return gradio_app


if __name__ == "__main__":
    ui()
