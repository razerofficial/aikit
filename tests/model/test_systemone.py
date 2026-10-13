"""SystemOne unit tests. No GPU or model: the wrapper talks to a fake vLLM over an in-process transport."""
import json
import math
import httpx
import pytest
from fastapi.testclient import TestClient

from rzr_aikit.utils.model_classifier import ModelCategory, classify_model
from vllm_plugins.systemone.backends import AutoTrustBackend, JevHead, MapikaBackend
from vllm_plugins.systemone.server import create_app
from vllm_plugins.systemone.decider import (
    DeciderConfig, answer_from_row_logprobs, chunk_letters, merge_letter_logprobs, plan_request,
)
from vllm_plugins.systemone.protocol import RequestError, parse_request

CHOICE = {"type": "choice", "instructions": "Which team?",
          "criteria": {"billing": "Charges", "technical": "Faults", "sales": "Buying"}}
SCORE = {"type": "score", "instructions": "How urgent?", "criteria": ["Routine", "Soon", "Today", "Now"]}
NOUL = {"type": "noul", "instructions": "Is there a receipt?"}
CFG = DeciderConfig.from_json({"temperature": 1.1, "temperature_by_type": {"choice": 1.164, "noul": 1.624, "score": 1.124}})


def body(questions, **over):
    return {"state": "ctx", "questions": questions, **over}


def plan(questions):
    return plan_request(parse_request(body(questions)))


def logprobs_for(probs, temperature):
    """Letter logprobs that softmax at `temperature` to exactly `probs`, up to a shared constant."""
    return [temperature * math.log(p) - 7.3 for p in probs]


# ---- request validation

def test_accepts_every_type_structured_state_and_the_limits():
    parse_request(body({"c": CHOICE, "s": SCORE, "n": NOUL}, state={"ticket": {"text": "hi"}}))
    for q in ({**CHOICE, "criteria": {f"o{i}": "d" for i in range(255)}}, {**SCORE, "criteria": ["x"] * 10}, {"type": "noul", "criteria": {"true": "it is"}}):
        parse_request(body({"q": q}))


@pytest.mark.parametrize("request_body, message", [
    ([], "object"),
    ({"questions": {"q": CHOICE}}, "state"),
    ({"state": "s"}, "questions"),
    (body({}), "questions"),
    (body({"q": {"type": "essay", "instructions": "x"}}), "questions.q"),
    (body({"q": {**CHOICE, "criteria": {"only": "one"}}}), "2..255"),
    (body({"q": {**CHOICE, "criteria": {f"o{i}": "d" for i in range(256)}}}), "2..255"),
    (body({"q": {**CHOICE, "instructions": ""}}), "instructions"),
    (body({"q": {**SCORE, "criteria": ["only"]}}), "2..10"),
    (body({"q": {**SCORE, "criteria": ["x"] * 11}}), "2..10"),
    (body({"q": {"type": "noul"}}), "noul"),
])
def test_rejects_invalid_requests_naming_the_problem(request_body, message):
    with pytest.raises(RequestError, match=message):
        parse_request(request_body)


# ---- rows and readout

def test_rows_per_type_and_independence_from_other_questions():
    rows = plan({"c": CHOICE, "s": SCORE, "n": NOUL}).rows
    assert [(len(r.options), r.answer_type) for r in rows] == [(3, "choice")] + [(2, "score")] * 4 + [(2, "noul")]
    alone = plan({"c": CHOICE}).rows[0]
    assert alone == [r for r in plan({"n": NOUL, "c": CHOICE, "s": SCORE}).rows if r.answer_type == "choice"][0]


def test_readout_recovers_choice_noul_and_score_distributions():
    choice = answer_from_row_logprobs(plan({"c": CHOICE}), [logprobs_for([0.6, 0.3, 0.1], 1.164)], CFG)["c"]
    assert choice["choice"] == "billing" and 0 <= choice["confidence"] <= 1
    assert list(choice["probabilities"].values()) == pytest.approx([0.6, 0.3, 0.1], abs=1e-3)

    noul = answer_from_row_logprobs(plan({"n": NOUL}), [logprobs_for([0.2, 0.8], 1.624)], CFG)["n"]
    assert noul["noul"] == pytest.approx(0.8, abs=1e-3)

    levels = [logprobs_for([1 - fit, fit], 1.124) for fit in (0.05, 0.1, 0.8, 0.05)]
    score = answer_from_row_logprobs(plan({"s": SCORE}), levels, CFG)["s"]
    assert max(score["probabilities"], key=score["probabilities"].get) == "2" and 1.5 < score["score"] < 2.5


def test_answers_keep_request_order_ignore_a_logprob_shift_and_check_the_row_count():
    p = plan({"n": NOUL, "c": CHOICE})
    out = answer_from_row_logprobs(p, [[-1.0, -2.0], [-1.0, -2.0, -3.0]], CFG)
    assert list(out) == ["n", "c"]
    assert out == answer_from_row_logprobs(p, [[-41.0, -42.0], [-41.0, -42.0, -43.0]], CFG)
    with pytest.raises(ValueError, match="rows"):
        answer_from_row_logprobs(p, [], CFG)


def test_config_defaults_and_chat_layouts_are_refused():
    assert (DeciderConfig.from_json({"temperature": 1.03}).for_type("score")) == 1.03  # decider-0.8b: no per-type map
    assert CFG.for_type("choice") == 1.164
    for chat in ({"layout": "chat"}, {"chat_template": True}):
        with pytest.raises(ValueError, match="chat"):
            DeciderConfig.from_json({"temperature": 1.0, **chat})
    for bad in ({"temperature": 0}, {"temperature": 1.0, "temperature_by_type": {"noul": -1}}):
        with pytest.raises(ValueError, match="positive"):
            DeciderConfig.from_json(bad)


@pytest.mark.parametrize("n, sizes", [(10, [10]), (128, [128]), (129, [128, 1]), (255, [128, 127])])
def test_letters_are_chunked_to_vllms_limit_and_merged_in_option_order(n, sizes):
    letters = list(range(1000, 1000 + n))
    chunks = chunk_letters(letters)
    assert [len(c) for c in chunks] == sizes
    assert merge_letter_logprobs(letters, [{t: -float(t) for t in c} for c in chunks]) == [-float(t) for t in letters]
    with pytest.raises(LookupError, match="1000"):
        merge_letter_logprobs(letters, [{}])


# ---- `rzr-aikit model run` wiring

@pytest.mark.parametrize("name, expected", [
    ("decider_config.json", ModelCategory.SYSTEMONE),
    ("adapter_vllm/decision_head.json", ModelCategory.SYSTEMONE),  # AutoTrust JEV
    ("decision_config.json", ModelCategory.STANDARD),  # Decision 2.0 is not supported yet
    ("config.json", ModelCategory.STANDARD),
    ("model_index.json", ModelCategory.STANDARD),
])
def test_local_models_are_classified_by_decider_config(tmp_path, name, expected):
    path = tmp_path / name
    path.parent.mkdir(exist_ok=True)
    path.write_text("{}")
    assert classify_model(str(tmp_path)) == expected


def test_diffusion_wins_over_systemone(tmp_path):
    (tmp_path / "model_index.json").write_text('{"_class_name": "P", "_diffusers_version": "1"}')
    (tmp_path / "decider_config.json").write_text("{}")
    assert classify_model(str(tmp_path)) == ModelCategory.DIFFUSION


@pytest.mark.parametrize("hub, expected", [
    (["config.json", "decider_config.json"], ModelCategory.SYSTEMONE),  # vLLM never downloads decider_config.json
    (["config.json"], ModelCategory.STANDARD),
    (OSError("offline"), ModelCategory.STANDARD),
])
def test_uncached_models_are_classified_from_the_hub_file_list(mocker, hub, expected):
    mocker.patch("rzr_aikit.utils.model_classifier._read_json_from_model", return_value=None)
    mocker.patch("rzr_aikit.utils.model_classifier._is_diffusion", return_value=False)
    api = mocker.patch("huggingface_hub.HfApi")
    if isinstance(hub, Exception):
        api.side_effect = hub
    else:
        api.return_value.model_info.return_value.siblings = [mocker.Mock(rfilename=n) for n in hub]
    assert classify_model("org/model") == expected


@pytest.mark.parametrize("system_one", [True, False])
def test_start_server_starts_the_wrapper_only_for_system_one_models(mocker, system_one):
    engine = mocker.Mock(omni=False, system_one=system_one, server_flags=["--lora-modules", "jev-decision=/x"] if system_one else [])
    engine.systemone_provider = "autotrust"
    engine.engine_args.model = engine.engine_args.served_model_name = "Mapika/decider-2b"
    engine.serve.return_value = mocker.Mock(poll=lambda: None)
    mocker.patch("builtins.open", mocker.mock_open(read_data="INFO: Starting vLLM API server\nINFO:     Application startup complete.\n"))
    popen = mocker.patch("subprocess.Popen")
    from rzr_aikit.model import run

    mocker.patch.object(run, "wait_for_systemone")

    printed = []
    mocker.patch.object(run.console, "print", side_effect=lambda *a, **k: printed.append(str(a[0])))
    run.start_server(engine, 8000, 8001)

    assert "env" not in engine.serve.call_args.kwargs and "VLLM_PLUGINS" not in str(engine.serve.call_args)
    assert popen.called is system_one
    assert engine.serve.call_args.kwargs["extra"] == ["--lora-modules", "jev-decision=/x"] if system_one else True
    if system_one:
        cmd = popen.call_args.args[0]
        assert "vllm_plugins.systemone.server" in cmd and cmd[cmd.index("--port") + 1] == "8001"
        assert cmd[cmd.index("--vllm-url") + 1] == "http://localhost:8000"
    assert any(("8001/v1/systemone" if system_one else "model generate") in line for line in printed)


def test_stop_also_finds_the_wrapper(mocker):
    from rzr_aikit.model import stop

    procs = [mocker.Mock(info={"cmdline": c}) for c in (["python", "-m", "vllm_plugins.systemone.server"], ["/x/bin/vllm", "serve"], ["bash"])]
    mocker.patch("psutil.process_iter", return_value=procs)
    assert stop.find_vllm_processes() == procs[:2]


# ---- wrapper HTTP layer, with a fake vLLM


class FakeVLLM:
    """Answers /v1/completions like vLLM: a logprob for every requested token id except `drop`."""

    def __init__(self, drop=None, down=False, status=200):
        self.drop, self.down, self.status, self.calls = drop, down, status, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("refused")
        if request.url.path == "/health":
            return httpx.Response(200)
        sent = json.loads(request.content)
        self.calls.append(sent)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "prompt is too long"}})
        top = {f"token_id:{t}": -1.0 - i for i, t in enumerate(sent["logprob_token_ids"]) if t != self.drop}
        return httpx.Response(200, json={"choices": [{"logprobs": {"top_logprobs": [top]}}]})


@pytest.fixture
def send():
    backend = MapikaBackend(CFG, tok=None, label_ids=list(range(1000, 1255)))
    backend.prompt = lambda row, state: [1, 2, 3]

    def post(vllm, **kwargs):
        client = httpx.AsyncClient(base_url="http://vllm", transport=httpx.MockTransport(vllm))
        app = create_app(backend, "http://vllm", "fake", client)
        return TestClient(app, raise_server_exceptions=False).post("/v1/systemone", **(kwargs or {"json": body({"q": CHOICE})}))

    return post


def test_valid_request_returns_probabilities_and_asks_vllm_for_the_option_tokens(send):
    vllm = FakeVLLM()
    response = send(vllm)
    assert response.status_code == 200 and response.json()["model"] == "fake"
    assert sum(response.json()["answers"]["q"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    call = vllm.calls[0]
    assert call["logprob_token_ids"] == call["allowed_token_ids"] == [1000, 1001, 1002]
    assert call["max_tokens"] == 1 and call["return_tokens_as_token_ids"] is True


def test_rows_wider_than_128_options_use_two_vllm_calls(send):
    wide = {**CHOICE, "criteria": {f"o{i}": "d" for i in range(129)}}
    vllm = FakeVLLM()
    assert send(vllm, json=body({"q": wide})).status_code == 200 and len(vllm.calls) == 2


@pytest.mark.parametrize("kwargs, vllm, status", [
    ({"content": b"{not json", "headers": {"Content-Type": "application/json"}}, FakeVLLM(), 422),
    ({"json": body({})}, FakeVLLM(), 422),
    ({}, FakeVLLM(drop=1001), 500),     # vLLM omitted a token: a server fault, reported as JSON
    ({}, FakeVLLM(down=True), 502),     # vLLM is not reachable
    ({}, FakeVLLM(status=400), 422),    # vLLM refused the input, for example a prompt past the context
])
def test_error_statuses(send, kwargs, vllm, status):
    response = send(vllm, **kwargs)
    assert response.status_code == status and response.json()["error"]["code"] == status


def test_vllm_input_errors_reach_the_client_with_vllms_message(send):
    assert "prompt is too long" in send(FakeVLLM(status=400)).json()["error"]["message"]


def test_run_refuses_a_systemone_port_equal_to_the_vllm_port(mocker):
    from rzr_aikit.model import run

    engine = mocker.Mock(omni=False, system_one=True)
    printed = []
    mocker.patch.object(run.console, "print", side_effect=lambda *a, **k: printed.append(str(a[0])))
    run.start_server(engine, 8001, 8001)
    assert not engine.serve.called and any("pick another" in line for line in printed)


def test_the_ready_message_waits_for_the_wrapper(mocker):
    from rzr_aikit.model import run

    wrapper = mocker.Mock(poll=lambda: 1)
    mocker.patch("builtins.open", mocker.mock_open(read_data="Address already in use\n"))
    with pytest.raises(RuntimeError, match="Address already in use"):
        run.wait_for_systemone(wrapper, 8001)
    up = mocker.patch("urllib.request.urlopen")
    run.wait_for_systemone(mocker.Mock(poll=lambda: None), 8001)
    assert "8001/health" in up.call_args.args[0]


def test_a_decider_found_on_the_hub_is_cached_for_offline_runs(mocker):
    mocker.patch("rzr_aikit.utils.model_classifier._read_json_from_model", return_value=None)
    api = mocker.patch("huggingface_hub.HfApi")
    api.return_value.model_info.return_value.siblings = [mocker.Mock(rfilename="decider_config.json")]
    download = mocker.patch("huggingface_hub.hf_hub_download")
    from rzr_aikit.utils.model_classifier import systemone_provider

    assert systemone_provider("org/decider") == "mapika"
    download.assert_called_once_with("org/decider", "decider_config.json")


# ---- AutoTrust JEV backend

HEAD = JevHead.from_json(
    {"verbalizer_ids": list(range(100, 124)), "bias": [0.1 * i for i in range(24)],
     "slots": {"ranges": {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]}}},
    {"per_kind": {"noul": 1.0143, "score": 1.0036, "choice": 1.0161}},
)
JEV = AutoTrustBackend(HEAD, tok=None, label_ids=list(range(1000, 1256)))
SCORE6 = {**SCORE, "criteria": [f"level {i}" for i in range(6)]}


def jev_plan(questions):
    return JEV.plan(parse_request(body(questions)))


def test_jev_prompt_and_option_tokens_per_kind():
    c, s, n = jev_plan({"c": CHOICE, "s": SCORE6, "n": NOUL}).rows
    assert JEV.prompt(c, "ctx") == "[kind] choice\n[state] ctx\n[question] Which team?\n[options]\nA) billing: Charges\nB) technical: Faults\nC) sales: Buying\n[decision]:"
    assert JEV.prompt(n, "ctx").endswith("[options]\nfalse\ntrue\n[decision]:")
    assert JEV.prompt(s, "ctx").endswith("[options]\n0\n1\n2\n3\n4\n5\n[decision]:")
    assert (JEV.options(c), JEV.options(s), JEV.options(n)) == ([1000, 1001, 1002], list(range(102, 108)), [100, 101])


def test_jev_readout_adds_the_slot_bias_and_divides_by_the_kind_temperature():
    plan = jev_plan({"c": CHOICE})
    lp = [-1.0, -2.0, -3.0]
    z = [(x + 0.1 * (8 + i)) / 1.0161 for i, x in enumerate(lp)]
    want = [math.exp(v) / sum(math.exp(w) for w in z) for v in z]
    got = JEV.answers(plan, [lp])["c"]["probabilities"]
    assert list(got.values()) == pytest.approx(want, abs=1e-3)
    assert plan.rows[0].answer_type == "choice"


def test_jev_choice_past_the_trained_slots_has_no_bias_and_uses_extended_labels():
    wide = {**CHOICE, "criteria": {f"o{i}": "d" for i in range(20)}}
    plan = jev_plan({"c": wide})
    assert JEV._slots(plan.rows[0])[1][16:] == [0.0] * 4 and "T) o19: d" in JEV.prompt(plan.rows[0], "x")
    assert len(JEV.options(plan.rows[0])) == 20


def test_jev_scores_only_on_six_levels():
    with pytest.raises(RequestError, match="exactly 6 levels"):
        jev_plan({"s": SCORE})
    assert jev_plan({"s": SCORE6}).rows[0].answer_type == "score"


def test_jev_completion_params_override_the_generation_config_truncation():
    assert JEV.completion_params([1, 2]) == {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "allowed_token_ids": [1, 2], "add_special_tokens": False}
    assert JEV.request_model == "jev-decision"


def test_jev_requests_go_to_the_lora_module_with_the_row_options():
    vllm = FakeVLLM()
    backend = AutoTrustBackend(HEAD, tok=None, label_ids=list(range(1000, 1256)))
    client = httpx.AsyncClient(base_url="http://vllm", transport=httpx.MockTransport(vllm))
    response = TestClient(create_app(backend, "http://vllm", "autotrust/JEV-9B", client)).post(
        "/v1/systemone", json=body({"c": CHOICE, "n": NOUL}))
    assert response.status_code == 200 and response.json()["model"] == "autotrust/JEV-9B"
    assert {c["model"] for c in vllm.calls} == {"jev-decision"}
    assert all(c["top_k"] == 0 and c["prompt"].endswith("[decision]:") for c in vllm.calls)


def test_jev_vllm_flags_register_the_lora_head_and_cap_sequences_for_the_multimodal_class(tmp_path):
    from rzr_aikit.model.run import systemone_vllm_flags

    (tmp_path / "adapter_vllm").mkdir()
    for arch, extra in (("Qwen3_5ForCausalLM", []), ("Qwen3_5ForConditionalGeneration", ["--mamba-cache-mode", "align", "--max-num-seqs", "8"])):
        (tmp_path / "config.json").write_text(json.dumps({"architectures": [arch]}))
        flags = systemone_vllm_flags("autotrust", str(tmp_path), [])
        assert flags[:7] == ["--enable-lora", "--max-lora-rank", "32", "--lora-modules", f"jev-decision={tmp_path}/adapter_vllm", "--logprobs-mode", "processed_logprobs"]
        assert flags[7:] == extra
    assert systemone_vllm_flags("mapika", "Mapika/decider-2b", []) == []
    assert "--max-num-seqs" not in systemone_vllm_flags("autotrust", str(tmp_path), ["--max-num-seqs", "4"])
