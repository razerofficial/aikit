"""SystemOne wrapper: `POST /v1/systemone` in front of a stock vLLM server.

    client --> this server (8001) --> vLLM /v1/completions (8000)

Each row is one completion request that generates a single token restricted to the row's option
tokens and reports their logprobs. A provider backend (MapikaBackend, ...) decides the prompt, the
option tokens and how the logprobs become answers.
"""
import argparse
import asyncio
import logging

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .backends import load_backend
from .decider import chunk_letters, merge_letter_logprobs
from .protocol import RequestError, parse_request

logger = logging.getLogger("vllm_plugins.systemone")


def create_app(backend, vllm_url: str, model: str, client: httpx.AsyncClient | None = None) -> FastAPI:
    """`model` is the name vLLM serves; it is also what answers report."""
    app = FastAPI()
    http = client or httpx.AsyncClient(base_url=vllm_url, timeout=600)

    async def option_logprobs(ids, tokens: list, chunk: list):
        """-> ({token id: logprob} for `chunk`, prompt tokens). `tokens` are all the row's option tokens: the
        engine restricts to them in every chunk, so the chunks share one normaliser."""
        r = await http.post("/v1/completions", json={
            "model": backend.request_model or model, "prompt": ids, "max_tokens": 1, "logprobs": 0,
            "logprob_token_ids": chunk, "return_tokens_as_token_ids": True, **backend.completion_params(tokens),
        })
        if 400 <= r.status_code < 500:  # for example a prompt longer than the served context
            raise Rejected(_vllm_message(r))
        r.raise_for_status()
        body = r.json()
        reported = body["choices"][0]["logprobs"]["top_logprobs"][0]
        found = {t: reported[f"token_id:{t}"] for t in chunk if f"token_id:{t}" in reported}
        return found, (body.get("usage") or {}).get("prompt_tokens", 0)

    async def score_row(row, state: str):
        """-> (option logprobs in option order, prompt token count)."""
        ids, tokens = backend.prompt(row, state), backend.options(row)
        results = await asyncio.gather(*(option_logprobs(ids, tokens, c) for c in chunk_letters(tokens)))
        merged = merge_letter_logprobs(tokens, [found for found, _ in results])
        return merged, len(ids) if isinstance(ids, list) else results[0][1]

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        try:
            req = parse_request(await request.json())
            plan = backend.plan(req)
        except (RequestError, ValueError) as e:  # a body that is not JSON is a ValueError
            return _error(422, str(e))
        state = backend.state(req.state)
        tasks = [asyncio.ensure_future(score_row(row, state)) for row in plan.rows]
        try:
            scored = await asyncio.gather(*tasks)
        except Rejected as e:
            _cancel(tasks)
            return _error(422, f"vLLM rejected the request: {e}")
        except httpx.TransportError as e:
            _cancel(tasks)
            return _error(502, f"vLLM at {vllm_url} is unreachable: {e}")
        except Exception as e:
            _cancel(tasks)  # do not leave the other rows' prefills running for a failed request
            logger.exception("SystemOne request failed")
            return _error(500, f"{type(e).__name__}: {e}")
        return JSONResponse({
            "model": model,
            "answers": backend.answers(plan, [lp for lp, _ in scored]),
            "usage": {"input_tokens": sum(n for _, n in scored), "output_tokens": 0},
        })

    @app.get("/health")
    async def health():
        try:
            (await http.get("/health")).raise_for_status()
        except httpx.HTTPError as e:
            return _error(502, f"vLLM at {vllm_url} is not healthy: {e}")
        return JSONResponse({"status": "ok"})

    return app


class Rejected(Exception):
    """vLLM refused a row because of the input (a 4xx)."""


def _vllm_message(r: httpx.Response) -> str:
    """vLLM's error text: {"error": {"message": ...}} or, for older routes, {"message": ...}."""
    try:
        body = r.json()
        error = body.get("error", body)
        return str(error.get("message", error)) if isinstance(error, dict) else str(error)
    except ValueError:
        return r.text[:300]


def _cancel(tasks: list) -> None:
    for task in tasks:
        task.cancel()


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "code": status}}, status_code=status)


def main(argv: list | None = None) -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="SystemOne wrapper for a running vLLM server")
    p.add_argument("--provider", choices=["mapika", "autotrust"], default="mapika")
    p.add_argument("--model", required=True, help="model path or Hub id (for its tokenizer and config)")
    p.add_argument("--served-model-name", help="the name vLLM serves the model under (default: --model)")
    p.add_argument("--vllm-url", default="http://localhost:8000")
    p.add_argument("--port", type=int, default=8001)
    a = p.parse_args(argv)
    app = create_app(load_backend(a.provider, a.model), a.vllm_url, a.served_model_name or a.model)
    uvicorn.run(app, host="0.0.0.0", port=a.port)


if __name__ == "__main__":
    main()
