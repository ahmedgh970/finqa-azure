import asyncio
import importlib.metadata
import json
import re
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai.types.responses import ResponseUsage

from ai103.agent.definition import INSTRUCTIONS
from ai103.agent.tools import TOOL_DEFINITIONS, Filing, ToolBox
from ai103.hosted import deploy
from ai103.hosted.app import respond
from ai103.hosted.chat import (
    CHAT_INSTRUCTIONS,
    AnswerStream,
    DirectRunner,
    answered_before,
    history_messages,
    render,
)
from ai103.retriever.core import Chunk

GOLDEN = Path("docs/financebench_open_source.jsonl")
CHUNK = Chunk(
    "c1",
    "Property, plant and equipment, net 8,738",
    "https://s/financial-pdfs/3M_2018_10K.pdf",
    1.0,
    "3M",
    2018,
    50,
    50,
)
FINAL = json.dumps(
    {
        "answer": "Net PP&E was 8,738 million [1].",
        "citations": [{"source": 1, "quote": "Property, plant and equipment, net 8,738"}],
        "insufficient_evidence": False,
    }
)
GENERAL = json.dumps(
    {
        "answer": "The quick ratio compares liquid assets to current liabilities.",
        "citations": [],
        "insufficient_evidence": False,
    }
)


class Item(SimpleNamespace):
    def model_dump(self, **_):
        return dict(vars(self))


def usage():
    return ResponseUsage.model_validate(
        {
            "input_tokens": 1000,
            "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 0},
            "output_tokens": 100,
            "output_tokens_details": {"reasoning_tokens": 10},
            "total_tokens": 1100,
        }
    )


def reply(output, text="", rid="resp"):
    return SimpleNamespace(id=rid, output=output, output_text=text, usage=usage())


def search_call():
    arguments = {"query": "net PP&E", "company": "3M", "fiscal_years": [2018], "k": 5}
    return Item(
        type="function_call",
        name="search_filings",
        call_id="call_1",
        arguments=json.dumps(arguments),
    )


def chat_runner(responses):
    client = Mock()
    snapshots = []

    def create(**kwargs):
        # La boucle complète la même liste à chaque tour : on garde ce qui a été envoyé.
        snapshots.append(list(kwargs["input"]))
        return responses.pop(0)

    client.responses.create.side_effect = create
    retriever = Mock()
    retriever.retrieve.return_value = [CHUNK]
    filings = [Filing("3M_2018_10K", "3M", 2018)]
    runner = DirectRunner(
        client, lambda: ToolBox(retriever, lambda q: [0.1] * 1024, filings), model="gpt-5.2"
    )
    return runner, client, snapshots


def test_direct_loop_keeps_history_and_tool_items_without_server_state(spans):
    runner, client, sent = chat_runner([reply([search_call()]), reply([], FINAL, rid="resp2")])
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! Ask me about the filings."},
    ]
    result = runner.answer("What is 3M net PP&E in FY2018?", history)
    assert result.status.value == "verified" and result.rounds == 2
    first = client.responses.create.call_args_list[0].kwargs
    assert first["instructions"] == CHAT_INSTRUCTIONS and first["store"] is False
    assert first["tools"] == TOOL_DEFINITIONS and "traceparent" in first["extra_headers"]
    assert sent[0] == [*history, {"role": "user", "content": "What is 3M net PP&E in FY2018?"}]
    # Deuxième tour : l'appel d'outil du modèle puis son résultat, renvoyés par notre code.
    assert [item.get("type") for item in sent[1][3:]] == ["function_call", "function_call_output"]
    chat = [s for s in spans.get_finished_spans() if s.name == "chat gpt-5.2"]
    assert len(chat) == 2 and chat[0].attributes["gen_ai.usage.cache_read.input_tokens"] == 200


def test_general_question_is_answered_without_tools_and_labelled():
    runner, client, _ = chat_runner([reply([], GENERAL)])
    result = runner.answer("What is a quick ratio?")
    assert result.tool_calls == [] and client.responses.create.call_count == 1
    text = render(result)
    assert text.startswith("The quick ratio") and "Réponse générale" in text


def test_reused_answer_is_labelled_and_its_figures_checked_against_the_conversation():
    reused = json.dumps(
        {
            "answer": "As said earlier, 3M's quick ratio was 0.85 in Q2 FY2023.",
            "citations": [],
            "insufficient_evidence": False,
            "basis": "conversation",
        }
    )
    runner, client, _ = chat_runner([reply([], reused), reply([], reused.replace("0.85", "0.91"))])
    history = [
        {"role": "user", "content": "3M quick ratio Q2 FY2023?"},
        {"role": "assistant", "content": "Le quick ratio de 3M est de 0,85 [1]."},
    ]
    ok = render(runner.answer("Same question?", history), history)
    assert "Reprise d'une réponse précédente" in ok and "chiffres retrouvés" in ok
    invented = render(runner.answer("Same question?", history), history)
    assert "chiffres absents de la conversation : 091" in invented
    assert client.responses.create.call_args.kwargs["text"]["format"]["schema"]["required"][-1] == (
        "basis"
    )


def test_render_lists_sources_checked_by_the_code():
    runner, _, _ = chat_runner([reply([search_call()]), reply([], FINAL)])
    text = render(runner.answer("3M PP&E?"))
    assert "[1] 3M_2018_10K, p. 50" in text and "(vérifié)" in text
    assert "extraits vérifiés" in text


def test_history_messages_keeps_text_turns_only():
    items = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
        {"type": "function_call", "name": "search_filings"},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Yo"}],
        },
        {"type": "message", "role": "user", "content": "Again"},
    ]
    assert history_messages(items) == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Yo"},
        {"role": "user", "content": "Again"},
    ]


def test_respond_runs_the_loop_off_the_event_loop():
    runner, _, _ = chat_runner([reply([], GENERAL)])
    history = [{"type": "message", "role": "user", "content": "Hello"}]
    text = asyncio.run(respond("What is a quick ratio?", history, runner))
    assert "quick ratio" in text


def test_chat_instructions_reuse_the_evaluated_method():
    assert CHAT_INSTRUCTIONS.endswith(INSTRUCTIONS.split("Method:\n", 1)[1])
    assert "Never state a company's reported figures from your own knowledge" in CHAT_INSTRUCTIONS
    assert "answer from that earlier answer without calling" in CHAT_INSTRUCTIONS


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden set local absent")
def test_prompts_contain_no_figure_from_the_golden_set():
    number = re.compile(r"\d[\d,.]*\d|\d")
    prompts = CHAT_INSTRUCTIONS + json.dumps(TOOL_DEFINITIONS)
    # Un nombre « significatif » : au moins 3 chiffres (hors puissances de 10) ou un séparateur.
    significant = {
        n
        for n in number.findall(prompts)
        if re.search(r"[.,]", n) or (len(n) >= 3 and n.strip("0") != "1")
    }
    answers = set()
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        answers |= set(number.findall(json.loads(line)["answer"]))
    assert significant and not significant & answers


def test_package_is_flat_and_contains_only_what_the_agent_needs(tmp_path):
    metadata = tmp_path / "meta.jsonl"
    metadata.write_text('{"doc_name": "3M_2018_10K", "company": "3M", "fiscal_year": 2018}\n')
    sha = deploy.package(tmp_path / "agent.zip", metadata)
    names = zipfile.ZipFile(tmp_path / "agent.zip").namelist()
    assert {"main.py", "requirements.txt", "corpus-metadata.jsonl"} <= set(names)
    assert "ai103/hosted/app.py" in names and "ai103/agent/tools.py" in names
    assert not any("__pycache__" in n or n.startswith("ai103/evaluation") for n in names)
    assert "ai103/hosted/regression.py" not in names
    assert len(sha) == 64
    requirements = zipfile.ZipFile(tmp_path / "agent.zip").read("requirements.txt").decode()
    assert "azure-ai-agentserver-responses==" in requirements


def test_requirements_pin_transitive_dependencies():
    # `core` n'est requis qu'indirectement (par `responses`) : sans épinglage, le build distant
    # a pris une version incompatible le 6 novembre.
    lines = deploy.requirements().splitlines()
    assert (
        "azure-ai-agentserver-core==" + importlib.metadata.version("azure-ai-agentserver-core")
        in lines
    )
    names = [line.split("==")[0] for line in lines]
    assert all("==" in line for line in lines) and names == sorted(set(names))


def test_package_entries_are_readable_regular_files_and_reproducible(tmp_path):
    metadata = tmp_path / "meta.jsonl"
    metadata.write_text("{}\n")
    first = deploy.package(tmp_path / "a.zip", metadata)
    for entry in zipfile.ZipFile(tmp_path / "a.zip").infolist():
        mode = entry.external_attr >> 16
        assert stat.S_ISREG(mode) and mode & 0o444 == 0o444, entry.filename
    assert deploy.package(tmp_path / "b.zip", metadata) == first


def test_definition_has_no_secret_and_uses_the_smallest_sandbox(monkeypatch):
    for name in deploy.ENVIRONMENT:
        monkeypatch.setenv(name, f"https://{name.lower()}.example")
    monkeypatch.setenv("AZURE_TOKEN_CREDENTIALS", "AzureCliCredential")
    document = deploy.definition_document(deploy.environment())
    assert (document["cpu"], document["memory"]) == ("0.5", "1Gi")
    assert document["code_configuration"]["runtime"] == "python_3_13"
    assert document["session_configuration"] == {"idle_timeout_seconds": 600}
    assert "AZURE_TOKEN_CREDENTIALS" not in document["environment_variables"]
    assert not any(
        word in name for name in document["environment_variables"] for word in ("KEY", "SECRET")
    )


def test_server_passes_the_conversation_history_on_the_next_turn(monkeypatch, tmp_path):
    import httpx

    from ai103.hosted import app as server

    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
    seen = []

    async def fake_stream(question, history, agent):
        seen.append(history_messages(history))
        # Deux morceaux : le client reçoit le texte complet, qu'il streame ou non.
        yield "Réponse à : "
        yield question

    monkeypatch.setattr(server, "stream_answer", fake_stream)
    monkeypatch.setattr(server, "runner", lambda: None)
    app = server.create_app(observability=False)

    async def conversation():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://local") as client:
            first = (await client.post("/responses", json={"input": "Bonjour"})).json()
            second = await client.post(
                "/responses", json={"input": "Et ensuite ?", "previous_response_id": first["id"]}
            )
            return second.json()

    second = asyncio.run(conversation())
    assert second["output"][0]["content"][0]["text"] == "Réponse à : Et ensuite ?"
    assert seen == [
        [],
        [
            {"role": "user", "content": "Bonjour"},
            {"role": "assistant", "content": "Réponse à : Bonjour"},
        ],
    ]


VERIFIED_REPLY = (
    "Le quick ratio de 3M est de 0,85 [1].\n\n_Statut : réponse sourcée, extraits vérifiés._"
)


def test_repeated_question_already_verified_forbids_tools_for_the_turn():
    reused = json.dumps(
        {
            "answer": "Comme indiqué plus haut, 0,85.",
            "citations": [],
            "insufficient_evidence": False,
            "basis": "conversation",
        }
    )
    runner, client, _ = chat_runner([reply([], reused)])
    history = [
        {"role": "user", "content": "Quel est le quick ratio de 3M au T2 2023 ?"},
        {"role": "assistant", "content": VERIFIED_REPLY},
    ]
    result = runner.answer("quel est le quick ratio de 3M au T2 2023", history)
    assert client.responses.create.call_args.kwargs["tool_choice"] == "none"
    assert "chiffres retrouvés dans la conversation" in render(result, history)


def test_failed_or_different_previous_answer_keeps_tools_available():
    failed = "Stopped: round_limit.\n\n_Statut : arrêt sur une limite (budget, tours ou coût)._"
    question = "Quel est le quick ratio de 3M au T2 2023 ?"
    assert not answered_before(
        question, [{"role": "user", "content": question}, {"role": "assistant", "content": failed}]
    )
    assert not answered_before(
        "Quel est le current ratio de Pfizer en 2021 ?",
        [{"role": "user", "content": question}, {"role": "assistant", "content": VERIFIED_REPLY}],
    )
    runner, client, _ = chat_runner([reply([], GENERAL)])
    runner.answer("What is a quick ratio?")
    assert client.responses.create.call_args.kwargs["tool_choice"] == "auto"


class FakeStream:
    """Flux d'événements Responses, utilisable avec `with … as events`."""

    def __init__(self, *events):
        self.events = events

    def __enter__(self):
        return iter(self.events)

    def __exit__(self, *exc):
        return False


def delta(text):
    return SimpleNamespace(type="response.output_text.delta", delta=text)


def completed(response):
    return SimpleNamespace(type="response.completed", response=response)


def test_chat_mode_chains_each_answer_to_the_previous_one(monkeypatch, capsys):
    client = Mock()
    client.responses.create.side_effect = [
        FakeStream(delta("Réponse "), delta(str(i)), completed(SimpleNamespace(id=f"resp_{i}")))
        for i in range(3)
    ]
    monkeypatch.setattr(deploy, "agent_client", lambda args: client)
    questions = iter(["Bonjour", "Et ensuite ?", "/nouveau", "Autre sujet", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(questions))
    deploy.chat(SimpleNamespace())
    # Champ absent (et non `null`) au début d'une conversation : Foundry refuse `null`.
    calls = client.responses.create.call_args_list
    previous = [c.kwargs.get("previous_response_id", "absent") for c in calls]
    assert previous == ["absent", "resp_0", "absent"]
    assert all(c.kwargs["stream"] is True for c in calls)
    out = capsys.readouterr().out
    assert "Nouvelle conversation" in out and "Réponse 1" in out and "premier texte" in out


def test_answer_stream_extracts_the_answer_field_whatever_the_split():
    answer = 'Ligne 1\nCafé "net" 😀 [1]'
    # ensure_ascii : é et l'emoji deviennent é et une paire 😀, coupables partout.
    payload = json.dumps({"answer": answer, "citations": [], "insufficient_evidence": False})
    for size in range(1, len(payload) + 1):
        stream = AnswerStream()
        shown = "".join(stream.feed(payload[i : i + size]) for i in range(0, len(payload), size))
        assert shown == answer, size
        assert stream.done


def streaming_runner(responses):
    """Le client renvoie un flux d'événements quand `stream=True`, la réponse sinon."""
    runner, client, sent = chat_runner(responses)
    plain = client.responses.create.side_effect

    def create(**kwargs):
        response = plain(**kwargs)
        if not kwargs.get("stream"):
            return response
        text = response.output_text
        return FakeStream(
            *(delta(text[i : i + 5]) for i in range(0, len(text), 5)), completed(response)
        )

    client.responses.create.side_effect = create
    return runner, client, sent


def test_direct_runner_streams_model_text_only_when_asked():
    runner, client, _ = streaming_runner([reply([search_call()]), reply([], FINAL, rid="r2")])
    received = []
    result = runner.answer("What is 3M net PP&E in FY2018?", on_text=received.append)
    assert "".join(received) == FINAL and result.status.value == "verified"
    assert all(c.kwargs["stream"] is True for c in client.responses.create.call_args_list)
    # Sans rappel, aucun streaming : campagnes et évaluations inchangées.
    runner, client, _ = streaming_runner([reply([], GENERAL)])
    runner.answer("What is a quick ratio?")
    assert "stream" not in client.responses.create.call_args.kwargs


def test_stream_answer_shows_the_answer_first_then_the_same_text_as_respond():
    from ai103.hosted import app as server

    async def collect(agent):
        return [chunk async for chunk in server.stream_answer("Net PP&E of 3M?", [], agent)]

    streaming, _, _ = streaming_runner([reply([search_call()]), reply([], FINAL, rid="r2")])
    chunks = asyncio.run(collect(streaming))
    plain, _, _ = chat_runner([reply([search_call()]), reply([], FINAL, rid="r2")])
    assert "".join(chunks) == asyncio.run(respond("Net PP&E of 3M?", [], plain))
    # La réponse arrive en plusieurs morceaux, puis sources et statut en un seul, à la fin.
    assert len(chunks) > 2 and "".join(chunks[:-1]) == "Net PP&E was 8,738 million [1]."


def test_stream_answer_appends_the_final_text_when_it_differs_from_what_was_shown(monkeypatch):
    from ai103.hosted import app as server

    class Agent:
        def answer(self, question, messages, on_text):
            on_text('{"answer": "Début')
            return None

    monkeypatch.setattr(server, "render", lambda result, messages: "Arrêt sur une limite.")

    async def collect():
        return "".join([c async for c in server.stream_answer("Q ?", [], Agent())])

    assert asyncio.run(collect()) == "Début\n\n---\n\nArrêt sur une limite."


def test_server_streams_text_deltas_over_sse(monkeypatch, tmp_path):
    import httpx

    from ai103.hosted import app as server

    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))

    async def fake_stream(question, history, agent):
        for chunk in ("Net PP&E ", "was 8,738 [1].", "\n\n_Statut : …_"):
            yield chunk

    monkeypatch.setattr(server, "stream_answer", fake_stream)
    monkeypatch.setattr(server, "runner", lambda: None)
    app = server.create_app(observability=False)

    async def call():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://local") as client:
            response = await client.post("/responses", json={"input": "PP&E ?", "stream": True})
            return response.text

    body = asyncio.run(call())
    deltas = [
        json.loads(line[5:])["delta"]
        for line in body.splitlines()
        if line.startswith("data:") and '"response.output_text.delta"' in line
    ]
    assert deltas == ["Net PP&E ", "was 8,738 [1].", "\n\n_Statut : …_"]
    assert "response.completed" in body
