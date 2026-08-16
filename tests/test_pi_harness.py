"""Tests for stock Pi harness selection, tracing, and sandbox construction."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchkit._pi_proxy import _capture_scaffold, _models_payload, _upstream_target
from benchkit.cli import _headless_jobs, _parse_args
from benchkit.client import _openai_metrics
from benchkit.engine import Engine, JobSpec, annotate_harness_effect
from benchkit.evaluation import EvaluationResult
from benchkit.pi_agent import PiAgentRunner, _RpcTrace
from benchkit.sandbox import (
    PI_DOCKERFILE,
    PI_PACKAGE,
    DockerTaskEnvironment,
    LatestPiImage,
    _docker_upstream,
    cleanup_owned_resources,
)


class LatestPiImageTests(unittest.TestCase):
    def test_generic_pi_sandbox_keeps_the_restricted_pid_limit(self) -> None:
        self.assertEqual(LatestPiImage(docker="docker").pids_limit, 256)

    def test_package_deliberately_tracks_npm_latest(self) -> None:
        self.assertEqual(PI_PACKAGE, "@earendil-works/pi-coding-agent@latest")
        self.assertIn(f"npm install -g {PI_PACKAGE}", PI_DOCKERFILE)

    def test_prepare_pulls_and_bypasses_build_cache_once_per_run(self) -> None:
        image = LatestPiImage(docker="docker")

        with patch("benchkit.sandbox._run") as run:
            run.return_value = SimpleNamespace(stdout="0.83.0\n")
            self.assertEqual(image.prepare(), "0.83.0")
            self.assertEqual(image.prepare(), "0.83.0")

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0][:2], ["docker", "version"])
        build = commands[1]
        self.assertEqual(build[:2], ["docker", "build"])
        self.assertIn("--pull", build)
        self.assertIn("--no-cache", build)
        self.assertEqual(
            commands[2],
            [
                "docker",
                "run",
                "--rm",
                "benchkit-pi:latest",
                "pi",
                "--version",
            ],
        )
        self.assertEqual(len(commands), 3)

    def test_cleanup_removes_the_transient_image(self) -> None:
        image = LatestPiImage(
            docker="docker",
            image="benchkit-pi:latest",
            version="0.83.0",
        )
        image._ready = True

        with patch("benchkit.sandbox._run") as run:
            image.cleanup()

        run.assert_called_once_with(
            ["docker", "image", "rm", "--force", "benchkit-pi:latest"],
            timeout=60,
        )
        self.assertFalse(image._ready)
        self.assertEqual(image.version, "")

    def test_cleanup_removes_every_resource_owned_by_the_runner(self) -> None:
        responses = [
            SimpleNamespace(stdout="container-one\ncontainer-two\n"),
            SimpleNamespace(stdout=""),
            SimpleNamespace(stdout="network-one\n"),
            SimpleNamespace(stdout=""),
        ]
        with patch("benchkit.sandbox._run", side_effect=responses) as run:
            cleanup_owned_resources("docker", "owner-123")

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            commands,
            [
                [
                    "docker",
                    "ps",
                    "--all",
                    "--quiet",
                    "--filter",
                    "label=benchkit.owner=owner-123",
                ],
                ["docker", "rm", "--force", "container-one", "container-two"],
                [
                    "docker",
                    "network",
                    "ls",
                    "--quiet",
                    "--filter",
                    "label=benchkit.owner=owner-123",
                ],
                ["docker", "network", "rm", "network-one"],
            ],
        )

    def test_local_inference_host_is_rewritten_for_docker(self) -> None:
        self.assertEqual(
            _docker_upstream("http://localhost:11434"),
            "http://host.docker.internal:11434/v1",
        )
        self.assertEqual(
            _docker_upstream("https://models.example.test/v1"),
            "https://models.example.test/v1",
        )

    def test_pi_starts_in_rpc_mode_without_replacing_native_tools(self) -> None:
        client = SimpleNamespace(host="http://local", api_key=None, provider="openai")
        image = LatestPiImage(docker="docker", version="test")
        environment = DockerTaskEnvironment(client, "model", image, docker="docker")
        environment._started = True

        with patch("benchkit.sandbox.subprocess.Popen") as popen:
            environment.start_pi()

        command = popen.call_args.args[0]
        self.assertEqual(
            command[-8:],
            [
                "pi",
                "--mode",
                "rpc",
                "--no-session",
                "--provider",
                "benchkit",
                "--model",
                "model",
            ],
        )
        self.assertNotIn("--tools", command)
        self.assertNotIn("--system-prompt", command)

    def test_runner_cleanup_sweeps_resources_and_removes_image(self) -> None:
        image = LatestPiImage(docker="docker")
        runner = PiAgentRunner(
            client=SimpleNamespace(),
            image=image,
            owner_id="owner-123",
        )

        with (
            patch("benchkit.pi_agent.cleanup_owned_resources") as resources,
            patch.object(image, "cleanup") as image_cleanup,
        ):
            runner.cleanup()

        resources.assert_called_once_with("docker", "owner-123")
        image_cleanup.assert_called_once_with()

    def test_pi_and_commands_can_run_in_task_specific_workdir(self) -> None:
        client = SimpleNamespace(host="http://local", api_key=None, provider="openai")
        image = LatestPiImage(docker="docker", version="test")
        environment = DockerTaskEnvironment(
            client,
            "model",
            image,
            docker="docker",
            workdir="/workspace/allergies",
        )
        environment._started = True

        with (
            patch("benchkit.sandbox.subprocess.Popen") as popen,
            patch("benchkit.sandbox._run") as run,
        ):
            environment.start_pi()
            environment.exec(["cmake", "-S", "."], workdir=environment.workdir)

        self.assertEqual(
            popen.call_args.args[0][3:5],
            ["--workdir", "/workspace/allergies"],
        )
        self.assertEqual(
            run.call_args.args[0][:5],
            [
                "docker",
                "exec",
                "--workdir",
                "/workspace/allergies",
                environment.container_name,
            ],
        )

    def test_task_container_has_no_host_mount_or_direct_egress(self) -> None:
        client = SimpleNamespace(
            host="http://localhost:11434",
            api_key="real-secret",
            provider="openai",
            context_length=lambda _model: 32768,
        )
        image = LatestPiImage(docker="docker", version="test")
        environment = DockerTaskEnvironment(
            client, "selected/model", image, docker="docker"
        )

        with patch("benchkit.sandbox._run") as run:
            run.return_value = SimpleNamespace(stdout="")
            environment.start()

        calls = run.call_args_list
        network = calls[0].args[0]
        proxy = calls[1].args[0]
        agent = calls[3].args[0]
        config = calls[4].kwargs["input_text"]
        self.assertEqual(network[:4], ["docker", "network", "create", "--internal"])
        for command in (network, proxy, agent):
            self.assertIn("benchkit.managed=true", command)
            self.assertIn(f"benchkit.owner={environment.owner_id}", command)
        self.assertIn("BENCHKIT_MODEL=selected/model", proxy)
        self.assertIn("BENCHKIT_UPSTREAM_API_KEY=real-secret", proxy)
        self.assertNotIn("BENCHKIT_UPSTREAM_API_KEY=real-secret", agent)
        self.assertIn(environment.network_name, agent)
        self.assertIn("no-new-privileges", agent)
        self.assertIn("ALL", agent)
        self.assertNotIn("--volume", agent)
        self.assertNotIn("--mount", agent)
        self.assertNotIn("/var/run/docker.sock", agent)
        self.assertEqual(
            json.loads(config)["providers"]["benchkit"]["models"],
            [
                {
                    "id": "selected/model",
                    "name": "selected/model",
                    "contextWindow": 32768,
                }
            ],
        )


class InferenceProxyTests(unittest.TestCase):
    def test_proxy_captures_exact_system_prompt_and_available_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "scaffold.json"
            with patch("benchkit._pi_proxy.SCAFFOLD_PATH", str(destination)):
                _capture_scaffold(
                    {
                        "messages": [
                            {"role": "system", "content": "Stock Pi prompt"},
                            {"role": "user", "content": "benchmark task"},
                        ],
                        "tools": [
                            {
                                "type": "function",
                                "function": {"name": "read"},
                            },
                            {
                                "type": "function",
                                "function": {"name": "bash"},
                            },
                        ],
                        "max_completion_tokens": 16384,
                    }
                )

            scaffold = json.loads(destination.read_text())

        self.assertEqual(scaffold["system_prompt"], "Stock Pi prompt")
        self.assertEqual(scaffold["system_prompt_chars"], 15)
        self.assertEqual(scaffold["tools_available"], ["read", "bash"])
        self.assertEqual(scaffold["max_output_tokens"], 16384)
        self.assertEqual(scaffold["max_output_tokens_field"], "max_completion_tokens")
        self.assertEqual(len(scaffold["system_prompt_sha256"]), 64)

    def test_proxy_exposes_only_the_selected_model(self) -> None:
        with patch.dict(os.environ, {"BENCHKIT_MODEL": "selected/model"}):
            payload = json.loads(_models_payload())

        self.assertEqual(payload["data"], [{"id": "selected/model", "object": "model"}])

    def test_proxy_preserves_base_and_request_queries(self) -> None:
        with patch.dict(
            os.environ,
            {"BENCHKIT_UPSTREAM": "https://models.example.test/v1?api-version=1"},
        ):
            target = _upstream_target("/v1/chat/completions?trace=yes")

        self.assertEqual(
            target,
            (
                "https",
                "models.example.test",
                443,
                "/v1/chat/completions?api-version=1&trace=yes",
            ),
        )


class PiRpcTraceTests(unittest.TestCase):
    def test_trace_measures_assistant_generation_separately(self) -> None:
        trace = _RpcTrace()

        with patch("benchkit.pi_agent.time.monotonic", side_effect=[10.0, 12.5]):
            trace.handle(
                {
                    "type": "message_start",
                    "message": {"role": "assistant"},
                }
            )
            trace.handle(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "answer"}],
                        "usage": {"output": 1},
                    },
                }
            )

        self.assertEqual(trace.generation_time_s, 2.5)

    def test_trace_keeps_native_tool_calls_and_all_turn_usage(self) -> None:
        trace = _RpcTrace()
        trace.handle(
            {
                "type": "tool_execution_start",
                "toolCallId": "call-1",
                "toolName": "bash",
                "args": {"command": "python3 solve.py"},
            }
        )
        trace.handle(
            {
                "type": "tool_execution_end",
                "toolCallId": "call-1",
                "toolName": "bash",
                "isError": False,
                "result": {
                    "content": [{"type": "text", "text": "42\n"}],
                },
            }
        )
        trace.handle(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "I should calculate."},
                        {"type": "text", "text": "The answer is 42."},
                    ],
                    "usage": {
                        "input": 100,
                        "cacheRead": 20,
                        "cacheWrite": 5,
                        "output": 25,
                    },
                    "stopReason": "stop",
                },
            }
        )

        self.assertEqual(trace.final_response, "The answer is 42.")
        self.assertEqual(trace.thinking_parts, ["I should calculate."])
        self.assertEqual((trace.input_tokens, trace.output_tokens), (125, 25))
        self.assertEqual(trace.turns, 1)
        self.assertEqual(trace.tool_trace[0]["name"], "bash")
        self.assertEqual(
            trace.tool_trace[0]["arguments"], {"command": "python3 solve.py"}
        )
        self.assertFalse(trace.tool_trace[0]["is_error"])
        self.assertEqual(trace.tool_trace[0]["output"], "42\n")
        self.assertEqual(trace.tool_calls_started, 1)

    def test_direct_metrics_keep_prompt_tokens_for_fair_cost_comparison(self) -> None:
        metrics = _openai_metrics(
            {
                "usage": {"prompt_tokens": 120, "completion_tokens": 30},
                "timings": {"predicted_ms": 1000},
            },
            1.2,
        )

        self.assertEqual(metrics["input_tokens"], 120)
        self.assertEqual(metrics["eval_count"], 30)

    def test_reasoning_only_length_stop_returns_partial_generation(self) -> None:
        events = [
            {
                "type": "message_start",
                "message": {"role": "assistant"},
            },
            {
                "type": "message_update",
                "assistantMessageEvent": {
                    "type": "thinking_delta",
                    "delta": "partial reasoning",
                },
            },
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [],
                    "usage": {"input": 1418, "output": 16384},
                    "stopReason": "length",
                },
            },
            {"type": "agent_settled"},
            {
                "id": "benchkit-stats-0",
                "type": "response",
                "data": {
                    "tokens": {"input": 1418, "output": 16384},
                    "assistantMessages": 1,
                },
            },
            {
                "id": "benchkit-final-0",
                "type": "response",
                "data": {"text": None},
            },
        ]

        class FakeProcess:
            stdin = io.StringIO()
            stdout = io.StringIO("".join(json.dumps(event) + "\n" for event in events))
            stderr = io.StringIO()

            @staticmethod
            def poll() -> int:
                return 0

        class FakeEnvironment:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self.process = FakeProcess()

            def __enter__(self):
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def start_pi(self) -> FakeProcess:
                return self.process

            @staticmethod
            def pi_scaffold() -> dict:
                return {}

        runner = PiAgentRunner(
            client=SimpleNamespace(),
            image=SimpleNamespace(
                version="0.84.1",
                prepare=Mock(return_value="0.84.1"),
                cleanup=Mock(),
            ),
        )

        with patch("benchkit.pi_agent.DockerTaskEnvironment", FakeEnvironment):
            generation = runner.generate("model", "question")

        self.assertTrue(generation["length_exceeded"])
        self.assertEqual(generation["done_reason"], "length")
        self.assertEqual(generation["response"], "")
        self.assertEqual(generation["thinking"], "partial reasoning")
        self.assertEqual(generation["eval_count"], 16384)
        self.assertEqual(generation["input_tokens"], 1418)
        self.assertGreater(generation["response_time_s"], 0)

    def test_scaffold_tokens_use_the_exact_captured_prompt(self) -> None:
        events = [
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "answer"}],
                    "usage": {"input": 10, "output": 1},
                    "stopReason": "stop",
                },
            },
            {"type": "agent_settled"},
            {
                "id": "benchkit-stats-0",
                "type": "response",
                "data": {
                    "tokens": {"input": 10, "output": 1},
                    "assistantMessages": 1,
                },
            },
            {
                "id": "benchkit-final-0",
                "type": "response",
                "data": {"text": "answer"},
            },
        ]

        class FakeProcess:
            stdin = io.StringIO()
            stdout = io.StringIO("".join(json.dumps(event) + "\n" for event in events))
            stderr = io.StringIO()

            @staticmethod
            def poll() -> int:
                return 0

        class FakeEnvironment:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self.process = FakeProcess()

            def __enter__(self):
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def start_pi(self) -> FakeProcess:
                return self.process

            @staticmethod
            def pi_scaffold() -> dict:
                return {
                    "system_prompt": "Exact stock Pi scaffold",
                    "system_prompt_sha256": "abc123",
                    "system_prompt_chars": 23,
                    "max_output_tokens": 16384,
                    "max_output_tokens_field": "max_completion_tokens",
                    "tools_available": ["read", "bash", "edit", "write"],
                }

        client = SimpleNamespace(tokenize=Mock(return_value=[1, 2, 3, 4, 5]))
        runner = PiAgentRunner(
            client=client,
            image=SimpleNamespace(
                version="0.84.1",
                prepare=Mock(return_value="0.84.1"),
                cleanup=Mock(),
            ),
        )

        with patch("benchkit.pi_agent.DockerTaskEnvironment", FakeEnvironment):
            generation = runner.generate("org/model", "question")

        client.tokenize.assert_called_once_with(
            "org/model", "Exact stock Pi scaffold", add_special=False
        )
        scaffold = generation["pi_scaffold"]
        self.assertEqual(scaffold["system_prompt_tokens"], 5)
        self.assertEqual(scaffold["system_prompt"], "Exact stock Pi scaffold")
        self.assertEqual(scaffold["system_prompt_chars"], 23)
        self.assertEqual(scaffold["system_prompt_sha256"], "abc123")
        self.assertEqual(scaffold["max_output_tokens"], 16384)
        self.assertEqual(scaffold["tools_available"], ["read", "bash", "edit", "write"])


class HarnessPairingTests(unittest.TestCase):
    def test_genuine_harness_errors_are_excluded_from_score(self) -> None:
        runner = Mock(version="test")
        runner.generate.side_effect = [
            {
                "thinking": "",
                "response": "pass",
                "trace_status": "unavailable",
                "tok_s": 10.0,
                "eval_count": 1,
                "eval_duration_ns": 100_000_000,
                "response_time_s": 0.1,
                "done_reason": "stop",
                "timed_out": False,
                "cancelled": False,
                "pi_scaffold": {
                    "system_prompt": "Stock Pi prompt",
                    "system_prompt_sha256": "abc123",
                    "system_prompt_chars": 15,
                    "system_prompt_tokens": 3,
                    "tools_available": ["read", "bash", "edit", "write"],
                },
            },
            RuntimeError("connection refused"),
        ]
        engine = Engine(
            client=SimpleNamespace(provider="openai", timeout=1.0),
            jobs=[JobSpec("model", "quickbench", "2", harness="pi")],
        )
        engine._pi_runner = runner

        with patch.object(
            engine,
            "_verify_response",
            return_value=EvaluationResult(score=1.0),
        ):
            result = engine.run()[0]

        self.assertEqual(result["passed"], 1)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["scored_total"], 1)
        self.assertEqual(result["harness_errors"], 1)
        self.assertEqual(result["score"], 100.0)
        self.assertEqual(result["tasks"][1]["outcome"], "harness_error")
        self.assertTrue(result["tasks"][1]["harness_error"])
        self.assertEqual(result["pi_system_prompt"], "Stock Pi prompt")
        self.assertEqual(
            result["pi_tools_available"], ["read", "bash", "edit", "write"]
        )

    def test_length_exceeded_is_a_scored_failure_with_partial_metrics(self) -> None:
        runner = Mock(version="test")
        runner.generate.side_effect = [
            {
                "thinking": "",
                "response": "pass",
                "trace_status": "unavailable",
                "tok_s": 10.0,
                "eval_count": 1,
                "eval_duration_ns": 100_000_000,
                "response_time_s": 0.1,
                "done_reason": "stop",
            },
            {
                "thinking": "truncated reasoning",
                "response": "",
                "trace_status": "observed",
                "tok_s": 27.3,
                "eval_count": 16384,
                "eval_duration_ns": 600_000_000_000,
                "response_time_s": 610.0,
                "done_reason": "length",
                "length_exceeded": True,
                "input_tokens": 1418,
            },
        ]
        engine = Engine(
            client=SimpleNamespace(provider="openai", timeout=1.0),
            jobs=[JobSpec("model", "quickbench", "2", harness="pi")],
        )
        engine._pi_runner = runner

        with patch.object(
            engine,
            "_verify_response",
            return_value=EvaluationResult(score=1.0),
        ):
            result = engine.run()[0]

        self.assertEqual(result["passed"], 1)
        self.assertEqual(result["scored_total"], 2)
        self.assertEqual(result["score"], 50.0)
        self.assertEqual(result["length_exceeded"], 1)
        self.assertEqual(result["harness_errors"], 0)
        self.assertEqual(result["throughput_items"], 2)
        task = result["tasks"][1]
        self.assertEqual(task["outcome"], "length_exceeded")
        self.assertTrue(task["length_exceeded"])
        self.assertFalse(task["harness_error"])
        self.assertEqual(task["output_tokens"], 16384)
        self.assertEqual(task["thinking"], "truncated reasoning")
        self.assertEqual(task["response_time_s"], 610.0)

    def test_engine_cleans_the_image_after_a_failed_pi_job(self) -> None:
        runner = Mock()
        engine = Engine(
            client=SimpleNamespace(),
            jobs=[JobSpec("model", "quickbench", "1", harness="pi")],
        )
        engine._pi_runner = runner

        with patch.object(engine, "_run_job", side_effect=RuntimeError("boom")):
            results = engine.run()

        self.assertEqual(results, [])
        runner.cleanup.assert_called_once_with()

    def test_cli_both_creates_matching_direct_and_pi_jobs(self) -> None:
        args = _parse_args(
            [
                "--headless",
                "--models",
                "model",
                "--benchmarks",
                "quickbench:1",
                "--harness",
                "both",
            ]
        )

        jobs = _headless_jobs(args, ["model"])

        self.assertEqual([job.harness for job in jobs], ["direct", "pi"])
        self.assertNotEqual(jobs[0].key, jobs[1].key)

    def test_pair_metrics_report_percentage_point_effect(self) -> None:
        common = {
            "model": "model",
            "benchmark": "quickbench",
            "variant": None,
            "slice": "2",
        }
        direct = {
            **common,
            "harness": "direct",
            "tasks": [
                {
                    "task_id": "a",
                    "score": 0.0,
                    "passed": False,
                    "loop_killed": True,
                },
                {"task_id": "b", "score": 100.0, "passed": True},
            ],
        }
        pi = {
            **common,
            "harness": "pi",
            "tasks": [
                {"task_id": "a", "score": 100.0, "passed": True},
                {"task_id": "b", "score": 100.0, "passed": True},
            ],
        }

        annotate_harness_effect([direct, pi])

        self.assertEqual(pi["harness_paired_total"], 2)
        self.assertEqual(pi["direct_score"], 50.0)
        self.assertEqual(pi["harness_score"], 100.0)
        self.assertEqual(pi["harness_score_delta_pp"], 50.0)
        self.assertEqual(pi["direct_loop_kill_rate"], 50.0)
        self.assertEqual(pi["harness_loop_kill_rate"], 0.0)
        self.assertEqual(pi["loop_kill_delta_pp"], -50.0)
        self.assertEqual(pi["harness_first_score_delta_pp"], 50.0)
        self.assertEqual(pi["harness_gains"], 1)
        self.assertEqual(pi["harness_regressions"], 0)

    def test_pair_metrics_separate_initial_and_repaired_harness_effects(self) -> None:
        common = {
            "model": "model",
            "benchmark": "quickbench",
            "variant": None,
            "slice": "1",
            "repair_attempts": 1,
        }
        direct = {
            **common,
            "harness": "direct",
            "tasks": [
                {
                    "task_id": "a",
                    "first_attempt_score": 0.0,
                    "score": 100.0,
                    "passed": True,
                }
            ],
        }
        pi = {
            **common,
            "harness": "pi",
            "tasks": [
                {
                    "task_id": "a",
                    "first_attempt_score": 100.0,
                    "score": 100.0,
                    "passed": True,
                }
            ],
        }

        annotate_harness_effect([direct, pi])

        self.assertEqual(pi["harness_first_score_delta_pp"], 100.0)
        self.assertEqual(pi["harness_score_delta_pp"], 0.0)

    def test_pair_metrics_drop_harness_errors_on_either_side(self) -> None:
        common = {
            "model": "model",
            "benchmark": "quickbench",
            "variant": None,
            "slice": "3",
        }
        direct = {
            **common,
            "harness": "direct",
            "tasks": [
                {"task_id": "a", "score": 100.0, "passed": True},
                {
                    "task_id": "b",
                    "score": 0.0,
                    "passed": False,
                    "timed_out": True,
                },
                {"task_id": "c", "score": 100.0, "passed": True},
            ],
        }
        pi = {
            **common,
            "harness": "pi",
            "tasks": [
                {
                    "task_id": "a",
                    "score": 0.0,
                    "passed": False,
                    "harness_error": True,
                },
                {
                    "task_id": "b",
                    "score": 0.0,
                    "passed": False,
                    "length_exceeded": True,
                },
                {"task_id": "c", "score": 100.0, "passed": True},
            ],
        }

        annotate_harness_effect([direct, pi])

        self.assertEqual(pi["harness_paired_total"], 2)
        self.assertEqual(pi["direct_score"], 50.0)
        self.assertEqual(pi["harness_score"], 50.0)


if __name__ == "__main__":
    unittest.main()
