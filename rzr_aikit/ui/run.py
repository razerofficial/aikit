import os
import sys
import subprocess
import threading
import typer
from typing_extensions import Annotated
from rich.console import Console
import time
import base64
import io
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import soundfile as sf

from rzr_aikit import ui_app

console = Console()


def _run_gradio_blocking(server_url: str, port: int):
    """Run Gradio server in blocking mode. Called inside the background subprocess."""
    import gradio as gr

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

    css = (
        " :has(> #prompt-textbox) { border: 0 !important; border-radius: 24px !important; }"
        " #ref-audio .audio-container { min-height: 175px !important; height: 175px !important; }"
        " #ref-audio .wrap { min-height: 0 !important; font-size: 0 !important; }"
        " #ref-audio .button { padding: 0px 4px !important; }"
        " #ref-audio .record-button { justify-content: center !important; }"
        " #ref-audio .mic-select { padding: 0px 4px !important; }"
        " #ref-audio .record-button::before { display: none !important; }"
        " #ref-audio .stop-button::before { display: none !important; }"
        " #ref-audio .stop-button-paused::before { display: none !important; }"
        " #ref-audio .stop-button { justify-content: center !important; }"
        " #ref-audio .stop-button-paused { justify-content: center !important; }"
        " #ref-audio .pause-button { padding:var(--spacing-sm) !important; }"
        " #ref-audio .resume-button { padding:var(--spacing-sm) !important; }"
    )
    gradio_app = create_gradio_app(server_url, theme=Razer, css=css)
    gradio_app.launch(
        server_port=port,
        server_name="0.0.0.0",
    )


@ui_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=True,
)
def run(
    server: Annotated[
        str,
        typer.Option(help="Server URL"),
    ] = "http://localhost:8000",
    port: Annotated[
        int,
        typer.Option(help="Gradio port"),
    ] = None,
):
    """
    Launch the UI for image generation.

    This command starts a local Gradio web UI for image generation and connects
    it to a running server endpoint. The Gradio server runs as a detached background
    process so the terminal session is not blocked. Use 'rzr-aikit ui stop' to stop it.

    Options:

        --server  Server URL to use for image generation requests
                  (default: http://localhost:8000)
        --port    Local port for the Gradio app
                  (default: 7860)

    Examples:

        $ rzr-aikit ui run
        $ rzr-aikit ui run --server http://127.0.0.1:8000
        $ rzr-aikit ui run --port 7861
    """
    from rzr_aikit.ui.stop import GRADIO_IDENTIFIER

    code = (
        f"# {GRADIO_IDENTIFIER}\n"
        f"from rzr_aikit.ui.run import _run_gradio_blocking\n"
        f"_run_gradio_blocking({repr(server)}, {port})"
    )

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    except Exception as e:
        console.print(f"Error launching UI server: {e}")
        return

    url_found = threading.Event()

    def watch_output():
        for line in process.stdout:
            stripped = line.strip()
            console.print(stripped)
            if "Running on" in stripped:
                time.sleep(1)
                url_found.set()
        process.stdout.close()

    t = threading.Thread(target=watch_output, daemon=True)
    t.start()

    with console.status("Starting UI server..."):
        url_found.wait(timeout=10)

    if process.poll() is not None:
        console.print("[red]UI server failed to start.[/red]")
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


def encode_audio_to_base64(audio_data: tuple) -> str:
    """Encode a Gradio audio tuple (sample_rate, numpy_array) to a base64 WAV data URL."""
    sample_rate, audio_np = audio_data
    if audio_np.dtype != np.int16:
        if audio_np.dtype in (np.float32, np.float64):
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_np = (audio_np * 32767).astype(np.int16)
        else:
            audio_np = audio_np.astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV")
    wav_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{wav_b64}"


def generate_audio(
    text: str,
    voice: str,
    language: str,
    instructions: str,
    ref_audio: tuple | None,
    ref_audio_url: str,
    ref_text: str,
    x_vector_only: bool,
    server_url: str,
) -> tuple | None:
    """Generate audio using the /v1/audio/speech API.

    Task type is auto-detected:
      - ref_audio or ref_audio_url provided → Base (Voice Clone)
      - instructions provided → VoiceDesign
      - otherwise → CustomVoice
    """
    import wave
    import gradio as gr

    if not text.strip():
        raise gr.Error("Please enter text to synthesize.")

    # Auto-detect task type
    has_ref = ref_audio is not None or bool(ref_audio_url and ref_audio_url.strip())
    if has_ref:
        task_type = "Base"
    elif instructions and instructions.strip():
        task_type = "VoiceDesign"
    else:
        task_type = "CustomVoice"

    model_name = None
    try:
        from rzr_aikit.utils.mlib import get_running_models

        models = [model.id for model in get_running_models()]
        model_name = models[0] if models else None
    except Exception:
        pass

    payload: dict = {
        "input": text.strip(),
        "response_format": "wav",
        "task_type": task_type,
    }
    if model_name:
        payload["model"] = model_name
    if language and language.strip():
        payload["language"] = language.strip()

    if task_type == "CustomVoice":
        if voice and voice.strip():
            payload["voice"] = voice.strip()
        if instructions and instructions.strip():
            payload["instructions"] = instructions.strip()
    elif task_type == "VoiceDesign":
        payload["instructions"] = instructions.strip()
    elif task_type == "Base":
        if ref_audio_url and ref_audio_url.strip():
            payload["ref_audio"] = ref_audio_url.strip()
        elif ref_audio is not None:
            payload["ref_audio"] = encode_audio_to_base64(ref_audio)
        if ref_text and ref_text.strip():
            payload["ref_text"] = ref_text.strip()
        if x_vector_only:
            payload["x_vector_only_mode"] = True

    try:
        response = requests.post(
            f"{server_url}/v1/audio/speech",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        with wave.open(BytesIO(response.content)) as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            raw = wav_file.readframes(n_frames)
        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype = dtype_map.get(sampwidth, np.int16)
        audio_np = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if sampwidth > 0:
            audio_np /= float(2 ** (8 * sampwidth - 1))
        if n_channels > 1:
            audio_np = audio_np.reshape(-1, n_channels)[:, 0]
        return sample_rate, audio_np
    except Exception as e:
        raise gr.Error(f"Audio generation failed: {e}")


def create_gradio_app(server_url: str, theme=None, css: str | None = None):
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
    from rzr_aikit.utils.mlib import get_running_models

    def get_current_model_html():
        """Get current running model name."""
        models = [model.id for model in get_running_models()]
        current_model = models[0] if len(models) > 0 else "No model running"
        return f'<h3 style="font-weight: 400;">Current model: {current_model}</h3>'

    def get_available_voices():
        """Get available voices for audio generation."""
        try:
            response = requests.get(f"{server_url}/v1/audio/voices", timeout=5)
            response.raise_for_status()
            data = response.json()
            voices = data.get("voices", [])
            return voices
        except Exception:
            return False

    voices = get_available_voices()

    with gr.Blocks(
        title="Razer Image Generation",
        theme=theme,
        css=css,
    ) as gradio_app:
        # Static header content
        gr.HTML("""
                <div align="left">
                <img src="https://iprototypes.com/grafana/pinkmanlogo.svg" width="900">
                </div>
                """)
        tab_heading = gr.HTML('<h1 style="margin-top: 5px; font-weight: 500;">Generate custom images with local models</h1>')
        model_display = gr.HTML()

        gradio_app.load(fn=get_current_model_html, outputs=model_display)

        with gr.Tabs() as tabs:
            with gr.Tab("Image Generation") as image_tab:
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
                                        value=9,
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
                                        value=0.0,
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
                    inputs=[
                        prompt,
                        negative_prompt,
                        height,
                        width,
                        steps,
                        cfg_scale,
                        seed,
                    ],
                )

            with gr.Tab("Audio Generation") as audio_tab:
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_text = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text to synthesize...",
                            lines=7,
                            elem_id="prompt-textbox",
                        )
                        audio_voice = gr.Dropdown(
                            choices=voices,
                            label="Voice",
                            value=voices[0] if voices else None,
                            allow_custom_value=True,
                        )
                        with gr.Accordion("Additional Settings", open=False):
                            audio_language = gr.Textbox(
                                label="Language",
                                placeholder="e.g. Auto, English, Chinese, Japanese",
                                lines=1,
                            )
                            audio_instructions = gr.Textbox(
                                label="Instructions / Voice Style",
                                placeholder="Style or emotion hints (e.g. 'speak with excitement'). If filled, Voice Design mode is used instead of a preset voice.",
                                lines=2,
                            )
                            gr.HTML(
                                "<p style='margin:8px 0 4px; font-size:0.85em; color:var(--body-text-color-subdued);'>"
                                "Voice Cloning — upload reference audio below to activate</p>"
                            )
                            ref_audio = gr.Audio(
                                label="Reference Audio",
                                type="numpy",
                                sources=["upload", "microphone"],
                                elem_id="ref-audio",
                            )
                            ref_audio_url = gr.Textbox(
                                label="Reference Audio URL",
                                placeholder="https://example.com/reference.wav",
                                lines=1,
                            )
                            ref_text = gr.Textbox(
                                label="Reference Transcript (optional)",
                                placeholder="Transcript of the reference audio — improves cloning quality",
                                lines=2,
                            )
                            x_vector_only = gr.Checkbox(
                                label="Use x-vector only (skip transcript)",
                                value=False,
                            )
                        audio_generate_btn = gr.Button(
                            "Generate Audio", variant="primary"
                        )
                    with gr.Column(scale=1):
                        audio_output = gr.Audio(
                            label="Generated Audio",
                            type="numpy",
                            autoplay=True,
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

        audio_generate_btn.click(
            fn=lambda t, v, lang, instr, ra, rau, rt, xv: generate_audio(
                t, v, lang, instr, ra, rau, rt, xv, server_url
            ),
            inputs=[
                audio_text,
                audio_voice,
                audio_language,
                audio_instructions,
                ref_audio,
                ref_audio_url,
                ref_text,
                x_vector_only,
            ],
            outputs=[audio_output],
        )

        audio_tab.select(fn=lambda: '<h1 style="margin-top: 5px; font-weight: 500;">Generate custom audio with local models</h1>', outputs=tab_heading)
        image_tab.select(fn=lambda: '<h1 style="margin-top: 5px; font-weight: 500;">Generate custom images with local models</h1>', outputs=tab_heading)

    return gradio_app


if __name__ == "__main__":
    run()
