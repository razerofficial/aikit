"""Provider backends for the SystemOne wrapper (server.py).

A backend turns a parsed request into scoring rows, builds each row's prompt, names the option tokens
to read, and turns the option logprobs back into answers. Each provider uses one protocol across its
models: Mapika Decider (0.8b, 2b, 4b) and AutoTrust JEV (9B, 27B, 27B-VL).
"""
import json
import math
import os
import string
from dataclasses import dataclass

from .decider import (
    DeciderConfig,
    Plan,
    Row,
    answer_from_row_logprobs,
    build_row_prompt,
    plan_request,
)
from .prompt import label_table
from .protocol import RequestError
from .systemone import assemble, render_state


def _local_or_hub(model: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    local = os.path.join(model, filename)
    return local if os.path.isfile(local) else hf_hub_download(model, filename)


def _read_json(model: str, filename: str) -> dict:
    with open(_local_or_hub(model, filename), encoding="utf-8") as f:
        return json.load(f)


class MapikaBackend:
    """Mapika Decider models: one row per question, letters after `Answer: (`."""

    request_model = None  # the served model
    state = staticmethod(render_state)

    def __init__(self, cfg: DeciderConfig, tok, label_ids: list):
        self.cfg, self.tok, self.label_ids = cfg, tok, label_ids

    @classmethod
    def load(cls, model: str) -> "MapikaBackend":
        from transformers import AutoTokenizer

        cfg = DeciderConfig.from_json(_read_json(model, "decider_config.json"))
        tok = AutoTokenizer.from_pretrained(model)
        return cls(cfg, tok, label_table(tok)[1])

    def plan(self, req) -> Plan:
        return plan_request(req)

    def prompt(self, row, state: str) -> list:
        return build_row_prompt(row, state, self.tok, self.cfg)

    def options(self, row) -> list:
        return self.label_ids[: len(row.options)]

    def completion_params(self, tokens: list) -> dict:
        return {"temperature": 0.0, "allowed_token_ids": tokens}

    def answers(self, plan, row_logprobs: list) -> dict:
        return answer_from_row_logprobs(plan, row_logprobs, self.cfg)


# JEV scores are fixed 0..5 and noul is false/true; only `choice` takes caller-defined options.
JEV_SCORE_LEVELS = 6
JEV_MAX_OPTIONS = 256


@dataclass(frozen=True)
class JevHead:
    """A JEV model's decision head (adapter_vllm/decision_head.json) and calibration.json."""

    ids: list      # verbalizer token ids, one per slot
    bias: list     # logit bias per slot
    ranges: dict   # kind -> [first slot, end slot)
    temps: dict    # kind -> temperature

    @classmethod
    def from_json(cls, head: dict, calibration: dict) -> "JevHead":
        slots = head["slots"]
        temps = {"noul": 1.0, "score": 1.0, "choice": 1.0, **calibration.get("per_kind", {})}
        return cls(list(head["verbalizer_ids"]), list(head["bias"]), slots["ranges"], temps)

    @property
    def trained_choices(self) -> int:
        lo, hi = self.ranges["choice"]
        return hi - lo


class AutoTrustBackend:
    """AutoTrust JEV models: one row per question, read at `[decision]:` through the `jev-decision` LoRA.

    noul reads the false/true slots, score the fixed 0-5 slots, choice the labels A-P and, past the
    trained slots, the extended single-pass labels A-Z, AA, ... (up to 256 options).
    """

    request_model = "jev-decision"  # the LoRA module that `rzr-aikit model run` registers

    def __init__(self, head: JevHead, tok, label_ids: list):
        self.head, self.tok, self.label_ids = head, tok, label_ids

    @classmethod
    def load(cls, model: str) -> "AutoTrustBackend":
        from transformers import AutoTokenizer

        head = JevHead.from_json(_read_json(model, "adapter_vllm/decision_head.json"), _read_json(model, "calibration.json"))
        tok = AutoTokenizer.from_pretrained(model)
        return cls(head, tok, _choice_label_ids(tok, head))

    def plan(self, req) -> Plan:
        """One row per question. A score question must use the six canonical 0-5 levels."""
        rows = []
        for qid, rq in req.rendered.items():
            if rq["type"] == "score" and len(rq["legend"]) != JEV_SCORE_LEVELS:
                raise RequestError(f"questions.{qid}: this model scores on exactly {JEV_SCORE_LEVELS} levels (0-5), got {len(rq['legend'])}")
            if rq["type"] == "choice" and len(rq["options"]) > JEV_MAX_OPTIONS:
                raise RequestError(f"questions.{qid}: choice takes at most {JEV_MAX_OPTIONS} options")
            rows.append(Row(rq["question"], list(rq["options"]), rq["type"]))
        index = [(qid, "list", i, 1) for i, qid in enumerate(req.rendered)]
        return Plan(req.rendered, index, rows)

    def state(self, state) -> str:
        return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)

    def prompt(self, row, state: str) -> str:
        if row.answer_type == "choice":
            lines = [f"{self.label(i)}) {o}" for i, o in enumerate(row.options)]
        elif row.answer_type == "score":
            lines = [str(i) for i in range(JEV_SCORE_LEVELS)]
        else:
            lines = ["false", "true"]
        return f"[kind] {row.answer_type}\n[state] {state}\n[question] {row.question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"

    def label(self, i: int) -> str:
        return _labels()[i]

    def _slots(self, row) -> tuple:
        lo, hi = self.head.ranges[row.answer_type]
        if row.answer_type == "choice":
            n = len(row.options)
            return self.label_ids[:n], [self.head.bias[lo + i] if lo + i < hi else 0.0 for i in range(n)]
        return self.head.ids[lo:hi], self.head.bias[lo:hi]

    def options(self, row) -> list:
        return self._slots(row)[0]

    def completion_params(self, tokens: list) -> dict:
        # The model's generation config sets top_k/top_p, which would truncate the processed logprobs.
        return {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "allowed_token_ids": tokens, "add_special_tokens": False}

    def answers(self, plan, row_logprobs: list) -> dict:
        if len(row_logprobs) != len(plan.rows):
            raise ValueError(f"expected {len(plan.rows)} rows of logprobs, got {len(row_logprobs)}")
        probs = []
        for row, logprobs in zip(plan.rows, row_logprobs):
            t, bias = self.head.temps[row.answer_type], self._slots(row)[1]
            z = [(lp + b) / t for lp, b in zip(logprobs, bias)]
            top = max(z)
            e = [math.exp(x - top) for x in z]
            probs.append([x / sum(e) for x in e])
        return assemble(plan.rendered, plan.index, probs)


def _labels() -> list:
    """A-Z, then AA, AB, ... (the single-pass labels JEV uses past its trained A-P slots)."""
    up = string.ascii_uppercase
    return list(up) + [a + b for a in up for b in up]


def _choice_label_ids(tok, head: JevHead) -> list:
    """Token ids of the option-line labels, as they tokenize after `\\n` and before `)`."""
    out = []
    for lab in _labels():
        ids = tok.encode(lab, add_special_tokens=False)
        if len(ids) == 1 and ids[0] in tok.encode(f"x\n{lab}) y", add_special_tokens=False):
            out.append(ids[0])
        if len(out) == JEV_MAX_OPTIONS:
            break
    lo, hi = head.ranges["choice"]
    if out[: hi - lo] != head.ids[lo:hi]:
        raise ValueError("the first choice labels do not match the trained A-P head")
    return out


def load_backend(provider: str, model: str):
    return {"mapika": MapikaBackend, "autotrust": AutoTrustBackend}[provider].load(model)
