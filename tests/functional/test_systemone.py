"""
Validates Notebook 12: SystemOne Decision Models

Manual functional test for a GPU host. It replays the notebook's workflow against a live server: the
three question types, several questions in one request, and routing on `confidence`. Serve a Decider
model in the background, then run this file:

    nohup rzr-aikit model run Mapika/decider-2b &        # wait for "SystemOne API ready" (port 8001)
    SYSTEMONE_MODEL=Mapika/decider-2b pytest tests/functional/test_systemone.py

Answers are checked against the upstream `decider-ai` 1.5.0 package (CPU, fp32) within TOL.
"""
import json
import os
import urllib.error
import urllib.request

import pytest

BASE = "http://localhost:8001"  # the SystemOne wrapper; vLLM itself stays on :8000
VLLM = "http://localhost:8000"
MODEL = os.environ.get("SYSTEMONE_MODEL", "Mapika/decider-2b")
AUTOTRUST = MODEL.startswith("autotrust/")
TOL = 1e-2

pytestmark = pytest.mark.llm
mapika_only = pytest.mark.skipif(AUTOTRUST, reason="Mapika Decider models only")

TEAM = {"type": "choice", "instructions": "Which team should handle this?", "criteria": {
    "billing": "Charges, invoices, payment problems",
    "technical_support": "Product setup and faults",
    "sales": "Questions about buying",
}}
URGENCY = {"type": "score", "instructions": "How urgent is this request?", "criteria": [
    "Routine: can wait a week", "Soon: this week", "Today: needs attention today", "Immediately: an outage, act now",
]}
URGENCY6 = {**URGENCY, "criteria": ["Not urgent", "Routine: can wait a week", "Soon: this week", "Today", "Immediately", "Critical: an outage, act now"]}
RECEIPT = {"type": "noul", "instructions": "Does the customer mention having a receipt?"}
STATES = {
    "choice": "My card was charged twice for the same purchase.",
    "score": "Our checkout is down for all customers right now and we are losing orders every minute.",
    "noul_yes": "Hi, I bought the blender last week and it stopped working. I have the receipt attached if you need it.",
    "noul_no": "Hi, my blender stopped working after a week. Can you send a replacement please?",
}
REQUESTS = {
    "choice": (STATES["choice"], {"q": TEAM}),
    "score": (STATES["score"], {"q": URGENCY}),
    "noul_yes": (STATES["noul_yes"], {"q": RECEIPT}),
    "noul_no": (STATES["noul_no"], {"q": RECEIPT}),
}
# Upstream decider-ai answers for REQUESTS: probabilities by option, or P(yes) for noul.
EXPECTED = {
    "Mapika/decider-0.8b": {
        "choice": {"billing": 0.9974, "technical_support": 0.0023, "sales": 0.0004},
        "score": {"0": 0.0012, "1": 0.0034, "2": 0.1878, "3": 0.8077}, "noul_yes": 0.9877, "noul_no": 0.0044,
    },
    "Mapika/decider-2b": {
        "choice": {"billing": 0.9677, "technical_support": 0.0200, "sales": 0.0123},
        "score": {"0": 0.0070, "1": 0.0989, "2": 0.2900, "3": 0.6042}, "noul_yes": 0.9467, "noul_no": 0.0664,
    },
    "Mapika/decider-4b": {
        "choice": {"billing": 0.9945, "technical_support": 0.0012, "sales": 0.0042},
        "score": {"0": 0.0026, "1": 0.0157, "2": 0.0372, "3": 0.9445}, "noul_yes": 0.9909, "noul_no": 0.1152,
    },
}


def post(state, questions):
    request = urllib.request.Request(
        BASE + "/v1/systemone", data=json.dumps({"state": state, "questions": questions}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as f:
            return f.status, json.load(f)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


@pytest.fixture(scope="module")
def server():
    try:
        urllib.request.urlopen(BASE + "/health", timeout=5)
    except OSError:
        pytest.skip(f"no SystemOne wrapper on :8001 (run: rzr-aikit model run {MODEL})")


# ---- live API

@mapika_only
@pytest.mark.parametrize("name", REQUESTS)
def test_answers_match_upstream(server, name):
    if MODEL not in EXPECTED:
        pytest.skip(f"no upstream reference values for {MODEL}")
    status, out = post(*REQUESTS[name])
    answer, want = out["answers"]["q"], EXPECTED[MODEL][name]
    assert status == 200 and out["model"] == MODEL
    assert out["usage"]["input_tokens"] > 0 and out["usage"]["output_tokens"] == 0
    if name.startswith("noul"):
        assert answer["noul"] == pytest.approx(want, abs=TOL)
        return
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert {k: answer["probabilities"][k] for k in want} == pytest.approx(want, abs=TOL)
    assert 0 <= answer["confidence"] <= 1


# The notebook's sample calls: (state, question, the answer a reader should see)
CHOICE_SAMPLES = [  # (state, instructions, the expected answer)
    ("My card was charged twice for the same purchase.", "Which team should handle this request?", "billing"),
    ("I have been paying for the basic plan for a year and I would like the one with more storage.", "What does the customer want?", "upgrade"),
    ("Thanks so much for sorting this out so quickly, you have been really helpful!", "What is the tone of this message?", "friendly"),
]
CHOICE_OPTIONS = {
    "billing": {"billing": "Charges, refunds and invoices", "technical_support": "Product faults and setup", "sales": "Questions about buying"},
    "upgrade": {"cancel": "The customer wants to end their subscription", "upgrade": "The customer wants a bigger or better plan",
                "question": "The customer is asking how something works", "complaint": "The customer is unhappy about something"},
    "friendly": {"angry": "Frustrated, upset or demanding", "neutral": "Matter of fact, no strong feeling", "friendly": "Warm, polite or grateful"},
}
SCORE_SAMPLES = [  # (state, instructions, levels, the range the expected score falls in)
    (STATES["score"], "How urgent is this request?", URGENCY["criteria"], (2.0, 3.0)),
    ("The headset is comfortable and the sound is great, but the microphone cuts out now and then.", "How positive is this review?",
     ["Very negative", "Negative", "Neutral", "Positive", "Very positive"], (1.5, 3.5)),
    ("Customer: How do I reset my password?\nAgent: Go to Settings, choose Security, then select Reset password. You will get an email with a link that works for one hour.",
     "How good is the agent's reply?", ["Unhelpful", "Weak", "Acceptable", "Good", "Excellent"], (2.0, 4.0)),
]
NOUL_SAMPLES = [  # (state, question, expected answer)
    (STATES["noul_yes"], "Does the customer mention having a receipt?", True),
    ("The jacket arrived in the wrong size. Please take it back and return my money.", "Is the customer asking for a refund?", True),
    ("Hello, I am still waiting for my parcel. It was meant to arrive on Monday.", "Does the message include an order number?", False),
]


@pytest.mark.parametrize("state, instructions, expected", CHOICE_SAMPLES)
@mapika_only
def test_notebook_choice_samples(server, state, instructions, expected):
    criteria = CHOICE_OPTIONS[expected]
    answer = post(state, {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}})[1]["answers"]["q"]
    assert answer["choice"] == expected and set(answer["probabilities"]) == set(criteria)


@pytest.mark.parametrize("state, instructions, levels, expected_range", SCORE_SAMPLES)
@mapika_only
def test_notebook_score_samples(server, state, instructions, levels, expected_range):
    answer = post(state, {"q": {"type": "score", "instructions": instructions, "criteria": levels}})[1]["answers"]["q"]
    assert expected_range[0] <= answer["score"] <= expected_range[1]
    assert list(answer["legend"].values()) == levels and len(answer["probabilities"]) == len(levels)


@pytest.mark.parametrize("state, question, expected", NOUL_SAMPLES)
@mapika_only
def test_notebook_noul_samples(server, state, question, expected):
    answer = post(state, {"q": {"type": "noul", "instructions": question}})[1]["answers"]["q"]
    assert (answer["noul"] > 0.5) is expected


@mapika_only
def test_low_confidence_answers_can_be_routed_to_a_person(server):
    """The notebook sends answers below a confidence threshold to a person."""
    team = {"type": "choice", "instructions": "Which team should handle this request?", "criteria": CHOICE_OPTIONS["billing"]}
    sure = post(STATES["choice"], {"q": team})[1]["answers"]["q"]["confidence"]
    unsure = post("I would like to talk to someone about my account.", {"q": team})[1]["answers"]["q"]["confidence"]
    assert sure >= 0.8 > unsure


def test_mixed_request_answers_each_key_in_order(server):
    status, out = post(STATES["score"], {"team": TEAM, "urgency": URGENCY6 if AUTOTRUST else URGENCY, "receipt": RECEIPT})
    assert status == 200
    assert [(k, a["type"]) for k, a in out["answers"].items()] == [("team", "choice"), ("urgency", "score"), ("receipt", "noul")]


def test_other_questions_and_option_order_do_not_change_an_answer(server):
    probs = lambda questions: post(STATES["choice"], questions)[1]["answers"]["q"]["probabilities"]
    urgency = URGENCY6 if AUTOTRUST else URGENCY
    alone = probs({"q": TEAM})
    reordered = {**TEAM, "criteria": dict(reversed(list(TEAM["criteria"].items())))}
    assert probs({"q": reordered}) == pytest.approx(alone, abs=TOL)
    assert probs({"q": TEAM, "other": RECEIPT, "urgency": urgency}) == pytest.approx(alone, abs=TOL)  # bf16 batching: up to 0.003


@pytest.mark.parametrize("n", [26, 27, 129, 255])
def test_wide_choice_questions(server, n):
    """Past 26 options the labels are two letters, and past 128 they need two engine calls."""
    criteria = {("billing" if i == 0 else f"option_{i}"): ("Refunds and charges" if i == 0 else f"Unrelated department {i}") for i in range(n)}
    status, out = post("The customer wants a refund for a broken blender.", {"q": {**TEAM, "criteria": criteria}})
    answer = out["answers"]["q"]
    assert status == 200 and answer["choice"] == "billing" and len(answer["probabilities"]) == n


@pytest.mark.parametrize("questions", [
    {},
    {"q": {"type": "essay", "instructions": "x"}},
    {"q": {**TEAM, "criteria": {f"o{i}": "d" for i in range(256)}}},
])
def test_invalid_requests_are_422_with_a_message(server, questions):
    status, out = post("s", questions)
    assert status == 422 and out["error"]["message"]


def test_vllm_stays_untouched_on_8000(server):
    with urllib.request.urlopen(VLLM + "/v1/models", timeout=30) as f:
        assert json.load(f)["data"]
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(urllib.request.Request(VLLM + "/v1/systemone", data=b"{}", headers={"Content-Type": "application/json"}))
    assert e.value.code == 404


# ---- prompt format against the real tokenizer (needs the model in the HF cache)

@mapika_only
def test_prompt_ends_at_the_answer_slot_and_hides_question_ids():
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    from vllm_plugins.systemone.decider import DeciderConfig, build_row_prompt, plan_request
    from vllm_plugins.systemone.protocol import parse_request

    with open(hf_hub_download("Mapika/decider-2b", "decider_config.json"), encoding="utf-8") as f:
        cfg = DeciderConfig.from_json(json.load(f))
    tok = AutoTokenizer.from_pretrained("Mapika/decider-2b")
    plan = plan_request(parse_request({"state": "My card was charged twice.", "questions": {"secret_id_123": TEAM}}))
    text = tok.decode(build_row_prompt(plan.rows[0], "My card was charged twice.", tok, cfg))
    assert text.endswith("Answer: (") and "secret_id_123" not in text


# ---- AutoTrust JEV: answers against JEV's own client (the model card's plain-vLLM recipe)
# Reference values from /tmp/sysone-up/jev_ref.py run against the same server (model "jev-decision" on :8000).
JEV_EXPECTED = {
    "autotrust/JEV-27B": {"choice": {"billing": 0.99996, "technical_support": 3e-05, "sales": 1e-05}, "score": {"0": 0.0001, "1": 0.0003, "2": 0.0003, "3": 0.0007, "4": 0.0962, "5": 0.9024}, "noul_yes": {"true": 0.9928}, "noul_no": {"true": 0.002}},
    "autotrust/JEV-27B-VL": {"choice": {"billing": 0.99997, "technical_support": 3e-05, "sales": 1e-05}, "score": {"0": 0.0001, "1": 0.0003, "2": 0.0002, "3": 0.0007, "4": 0.0962, "5": 0.9024}, "noul_yes": {"true": 0.9929}, "noul_no": {"true": 0.002}},
    "autotrust/JEV-9B": {"choice": {"billing": 0.9999, "technical_support": 0.0, "sales": 0.0001}, "score": {"0": 0.0001, "1": 0.0001, "2": 0.0003, "3": 0.0029, "4": 0.2591, "5": 0.7376}, "noul_yes": {"true": 0.9855}, "noul_no": {"true": 0.0124}},
}


autotrust_only = pytest.mark.skipif(not AUTOTRUST, reason="AutoTrust JEV models only")


@autotrust_only
@pytest.mark.parametrize("name", ["choice", "score", "noul_yes", "noul_no"])
def test_autotrust_answers_match_the_model_cards_client(server, name):
    if MODEL not in JEV_EXPECTED:
        pytest.skip(f"no reference values for {MODEL}")
    body = {"choice": ("choice", {"q": TEAM}), "score": ("score", {"q": URGENCY6}),
            "noul_yes": ("noul_yes", {"q": RECEIPT}), "noul_no": ("noul_no", {"q": RECEIPT})}[name]
    status, out = post(STATES[body[0]], body[1])
    answer, want = out["answers"]["q"], JEV_EXPECTED[MODEL][name]
    assert status == 200 and out["model"] == MODEL and out["usage"]["input_tokens"] > 0
    if name.startswith("noul"):
        assert answer["noul"] == pytest.approx(want["true"], abs=TOL)
    elif name == "score":
        assert [answer["probabilities"][str(i)] for i in range(6)] == pytest.approx([want[str(i)] for i in range(6)], abs=TOL)
    else:
        got = list(answer["probabilities"].values())
        assert got == pytest.approx(list(want.values()), abs=TOL)


@autotrust_only
def test_autotrust_scores_only_on_six_levels(server):
    status, out = post("s", {"q": URGENCY})
    assert status == 422 and "6 levels" in out["error"]["message"]
