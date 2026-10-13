"""Decider readout: SystemOne requests -> one scoring row per question -> answers.

Every question is scored in its own row (the state plus that question), so other questions never
change an answer. A score question becomes one yes/no row per level. Each row ends at the answer
slot `Answer: (`; the logprobs of its option letters, softmaxed at the model's per-type
temperature, are the probabilities.
"""
import math
from dataclasses import dataclass

from .prompt import MAX_OPTIONS, build, resolve_layout
from .systemone import assemble, plan_rows, row_types

# vLLM accepts at most this many `logprob_token_ids` per request (128 in vLLM 0.30).
MAX_LOGPROB_TOKEN_IDS = 128


@dataclass(frozen=True)
class DeciderConfig:
    """The fields of a model's decider_config.json that the readout uses."""

    temperature: float
    temperature_by_type: dict
    max_state_tokens: int

    @classmethod
    def from_json(cls, cfg: dict) -> "DeciderConfig":
        if resolve_layout(cfg) != "plain":  # also rejects contradictory configs
            raise ValueError(f"this plugin only supports the plain prompt layout, not {resolve_layout(cfg)!r}")
        temps = {k: float(v) for k, v in (cfg.get("temperature_by_type") or {}).items()}
        if any(t <= 0 for t in [float(cfg["temperature"]), *temps.values()]):
            raise ValueError("decider_config.json temperatures must be positive")
        return cls(
            temperature=float(cfg["temperature"]),
            temperature_by_type=temps,
            max_state_tokens=int(cfg.get("max_state_tokens", 32768)),
        )

    def for_type(self, answer_type: str) -> float:
        return self.temperature_by_type.get(answer_type, self.temperature)


@dataclass(frozen=True)
class Row:
    question: str
    options: list
    answer_type: str


@dataclass
class Plan:
    rendered: dict  # question id -> render_question() output
    index: list     # (id, "iso" | "list", first row, n rows), as assemble() expects
    rows: list


def plan_request(req) -> Plan:
    flat, index = plan_rows(req.rendered, isolated=True)
    types = row_types(req.rendered, index)
    return Plan(req.rendered, index, [Row(r["question"], list(r["options"]), t) for r, t in zip(flat, types)])


def answer_from_row_logprobs(plan: Plan, row_logprobs: list, cfg: DeciderConfig) -> dict:
    """row_logprobs[i] holds the option-letter logprobs of plan.rows[i], in option order."""
    if len(row_logprobs) != len(plan.rows):
        raise ValueError(f"expected {len(plan.rows)} rows of logprobs, got {len(row_logprobs)}")
    probs = []
    for row, logprobs in zip(plan.rows, row_logprobs):
        t, top = cfg.for_type(row.answer_type), max(logprobs)
        z = [math.exp((x - top) / t) for x in logprobs]
        probs.append([v / sum(z) for v in z])
    return assemble(plan.rendered, plan.index, probs)


class _Q:
    gold = 0

    def __init__(self, text, options):
        self.text, self.options = text, options


class _Example:
    def __init__(self, context, questions):
        self.context, self.qs = context, questions


class _KeepOrder:
    def shuffle(self, x):
        pass

    def sample(self, xs, k):
        return xs[:k]


def build_row_prompt(row: Row, state_text: str, tok, cfg: DeciderConfig) -> list:
    """Token ids for one row, ending at the answer slot `Answer: (`."""
    return build(
        _Example(state_text, [_Q(row.question, row.options)]),
        tok,
        _KeepOrder(),
        max_options=MAX_OPTIONS,
        max_ctx_tokens=cfg.max_state_tokens,
        layout="plain",
    )["ids"]


def chunk_letters(letters: list) -> list:
    """Split option-letter token ids into groups vLLM can report in one request."""
    n = MAX_LOGPROB_TOKEN_IDS
    return [letters[i : i + n] for i in range(0, len(letters), n)]


def merge_letter_logprobs(letters: list, results: list) -> list:
    """Combine per-chunk {token id: logprob} dicts into one list in option order."""
    merged = {tid: lp for got in results for tid, lp in got.items()}
    missing = [t for t in letters if t not in merged]
    if missing:
        raise LookupError(f"engine returned no logprob for option letter token ids {missing[:8]}")
    return [merged[t] for t in letters]
