"""The RLM strategies, with a stand-in for the authors' package (its own code runs in Docker)."""

from __future__ import annotations

import json
import types
import unittest
from typing import Any
from unittest import mock

from relay.core.engine import Engine
from relay.core.ir import Item, Kind, Request
from relay.harnesses import Codex, Harness
from relay.protocols import OpenAIResponses
from relay.strategies import RLM, PersistentRLM
from relay.strategies import rlm as module


class Runtime:
    """The authors' RLM as the strategy drives it: one model call per completion, then an answer."""

    made: list[Runtime] = []

    def __init__(self, **options: Any) -> None:
        self.options, self.contexts, self.closed = options, [], False
        self.backend_kwargs: dict[str, Any] = {}
        Runtime.made.append(self)

    def completion(self, context: Any, root_prompt: str) -> Any:
        self.contexts.append(context)
        self.prompt = root_prompt
        answer = self.backend_kwargs["client"].completion(
            [{"role": "system", "content": "RLM SYSTEM"}, {"role": "user", "content": "go"}])
        return types.SimpleNamespace(response=answer)

    def close(self) -> None:
        self.closed = True


class Model:
    def __init__(self) -> None:
        self.asked: list[list[Item]] = []

    def complete(self, items: list[Item]) -> str:
        self.asked.append(items)
        return "Run `cat file_2.txt`."


def request(*items: Item, conversation: str = "c") -> Request:
    return Request(items, items, frozenset(range(len(items) + 1)), 1_000, None, conversation=conversation)


TASK = Item(Kind.USER, "read the files", 1)
CALL, RESULT = Item(Kind.TOOL_CALL, 'sh({"cmd": "cat file_1.txt"})', 2), Item(Kind.TOOL_RESULT, "CODE: c1", 3)


class RLMTests(unittest.TestCase):
    def setUp(self) -> None:
        Runtime.made = []
        patcher = mock.patch.object(module, "_official", return_value=types.SimpleNamespace(RLM=Runtime))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_conversation_is_the_context_and_the_answer_the_handoff(self) -> None:
        model = Model()
        context = RLM().plan(request(TASK, CALL, RESULT), model)
        runtime, = Runtime.made
        self.assertEqual(runtime.contexts, [[{"role": "user", "content": "read the files"},
                                             {"role": "tool_call", "content": CALL.text},
                                             {"role": "tool", "content": "CODE: c1"}]])
        self.assertEqual(runtime.options["persistent"], False)
        self.assertEqual(model.asked, [[Item(Kind.SYSTEM, "RLM SYSTEM"), Item(Kind.USER, "go")]])  # the task's model
        handoff, = context.items
        self.assertIs(handoff.kind, Kind.SUMMARY)
        self.assertIn("Run `cat file_2.txt`.", handoff.text)
        RLM().plan(request(TASK, CALL, RESULT), model)
        self.assertEqual(len(Runtime.made), 2)  # a fresh query every request

    def test_persistent_adds_only_what_is_new_and_starts_over_when_the_history_changes(self) -> None:
        strategy, model = PersistentRLM(), Model()
        strategy.plan(request(TASK), model)
        strategy.plan(request(TASK, CALL, RESULT), model)
        runtime, = Runtime.made
        self.assertEqual(runtime.options["persistent"], True)
        self.assertEqual([len(c) for c in runtime.contexts], [1, 2])  # context_0, then context_1: the call and result
        self.assertIn("context_k", runtime.prompt)
        strategy.plan(request(Item(Kind.USER, "something else", 1)), model)  # not a continuation
        self.assertTrue(runtime.closed)
        self.assertEqual(len(Runtime.made), 2)
        strategy.plan(request(TASK, conversation="other"), model)
        self.assertEqual(len(strategy.sessions), 2)

    def test_persistent_repeats_latest_constraints_and_only_new_conversation_items(self) -> None:
        strategy, model = PersistentRLM(), Model()
        first = {
            "protocol": "openai_responses",
            "request": {"model": "m", "input": [], "tool_choice": "required"},
            "items": [
                {"type": "message", "role": "developer", "content": "cwd=/a; read-only"},
                {"type": "message", "role": "user", "content": "read the files"},
            ],
            "context_indices": [0],
        }
        second = {
            "protocol": "openai_responses",
            "request": {"model": "m", "input": [], "tool_choice": "auto"},
            "items": [
                {"type": "message", "role": "developer", "content": "cwd=/b; workspace-write"},
                {"type": "message", "role": "user", "content": "read the files"},
                {"type": "function_call", "call_id": "c1", "name": "sh", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "done"},
            ],
            "context_indices": [0],
        }

        model.codec, model.body = OpenAIResponses(), {**first["request"], "input": first["items"]}
        strategy.plan(request(TASK), model)
        model.body = {**second["request"], "input": second["items"]}
        strategy.plan(request(TASK, CALL, RESULT), model)

        runtime, = Runtime.made
        self.assertEqual(runtime.contexts, [
            first,
            {
                **second,
                "items": [second["items"][0], *second["items"][2:]],
                "context_indices": [0],
            },
        ])

    def test_persistent_keeps_its_runtime_when_the_engines_early_conversation_name_changes(self) -> None:
        codec, strategy = OpenAIResponses(), PersistentRLM()
        engine = Engine(strategy)

        def upstream(_: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            return 200, {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Take the next step."}]}]}

        initial = [
            {"type": "additional_tools", "role": "developer", "tools": [
                {"type": "function", "name": "sh", "parameters": {"type": "object"}}]},
            {"type": "message", "role": "user", "content":
             "<environment_context><cwd>/a</cwd></environment_context>"},
            {"type": "message", "role": "user", "content": "read the files"},
        ]
        later = [
            *initial,
            {"type": "function_call", "call_id": "c1", "name": "sh", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "done"},
        ]

        engine.prepare(codec, Codex(), {"model": "m", "input": initial}, tenant="t", post=upstream)
        engine.prepare(codec, Codex(), {"model": "m", "input": later}, tenant="t", post=upstream)

        runtime, = Runtime.made
        self.assertEqual(len(runtime.contexts), 2)
        self.assertEqual(runtime.contexts[1]["items"], [initial[0], initial[1], *later[3:]])
        self.assertEqual(runtime.contexts[1]["context_indices"], [0, 1])

    def test_tool_choice_none_bypasses_the_rlm_even_when_tools_are_listed(self) -> None:
        body = {
            "model": "m",
            "input": [{"type": "message", "role": "user", "content": "answer without tools"}],
            "tools": [{"type": "function", "name": "sh", "parameters": {"type": "object"}}],
            "tool_choice": "none",
        }

        exchange = Engine(RLM()).prepare(
            OpenAIResponses(), Harness(), body, tenant="t",
            post=lambda _: self.fail("the RLM model must not be called"),
        )

        self.assertEqual(Runtime.made, [])
        self.assertEqual(exchange.body, body)

    def test_through_the_engine_the_model_gets_the_rlms_own_request(self) -> None:
        codec, sent = OpenAIResponses(), []

        def upstream(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            sent.append(body)
            return 200, {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Run `cat file_2.txt`."}]}]}

        history = [{"type": "message", "role": "developer", "content": "harness rules"},
                   {"type": "message", "role": "user", "content": "read the files"}]
        body = {"model": "m", "instructions": "harness", "input": history, "tools": [{"type": "function", "name": "sh"}]}
        exchange = Engine(RLM()).prepare(codec, Harness(), body, tenant="t", post=upstream)
        rlm_call, = sent
        self.assertEqual(rlm_call["instructions"], "RLM SYSTEM")
        self.assertNotIn("tools", rlm_call)
        self.assertEqual([codec.classify(i).text for i in rlm_call["input"]], ["go"])
        forwarded = exchange.body
        self.assertEqual(forwarded["tools"], body["tools"])  # the step itself is the harness's model's, with its tools
        self.assertIn("Run `cat file_2.txt`.", json.dumps(forwarded["input"]))
        self.assertNotIn("read the files", json.dumps(forwarded["input"]))

    def test_the_rlm_gets_the_complete_protocol_request_as_external_data(self) -> None:
        codec, sent = OpenAIResponses(), []

        def upstream(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            sent.append(body)
            return 200, {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Run the tests."}]}]}

        tools = [{"type": "function", "name": "exec_command", "parameters": {
            "type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}]
        history = [
            {"type": "additional_tools", "role": "developer", "tools": tools},
            {"type": "message", "role": "developer", "content":
             "<permissions instructions>read only</permissions instructions>"},
            {"type": "message", "role": "user", "content":
             "# AGENTS.md instructions\n\n<INSTRUCTIONS>test first</INSTRUCTIONS>"},
            {"type": "message", "role": "user", "content": "inspect the failure"},
            {"type": "function_call", "call_id": "call_1", "name": "exec_command",
             "arguments": '{"cmd":"sed -n 1,80p app.py"}'},
            {"type": "function_call_output", "call_id": "call_1", "output": "missing check"},
        ]
        body = {"model": "m", "instructions": "stay read-only", "input": history,
                "tool_choice": "required", "parallel_tool_calls": False, "store": False}

        Engine(RLM()).prepare(codec, Codex(), body, tenant="t", post=upstream)
        runtime, = Runtime.made
        self.assertEqual(runtime.contexts, [{
            "protocol": "openai_responses",
            "request": {**body, "input": []},
            "items": history,
            "context_indices": [0, 1, 2],
        }])


if __name__ == "__main__":
    unittest.main()
