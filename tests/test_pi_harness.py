"""Tests for stock Pi harness selection, tracing, and sandbox construction."""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchkit._pi_proxy import _models_payload, _upstream_target
from benchkit.cli import _headless_jobs, _parse_args
from benchkit.client import _openai_metrics
from benchkit.engine import Engine, JobSpec, annotate_harness_effect
from benchkit.pi_agent import _RpcTrace
from benchkit.sandbox import (
    PI_DOCKERFILE,
    PI_PACKAGE,
    DockerTaskEnvironment,
    LatestPiImage,
    _docker_upstream,
)


class LatestPiImageTests(unittest.TestCase):
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
                    "usage": {"input": 100, "output": 25},
                    "stopReason": "stop",
                },
            }
        )

        self.assertEqual(trace.final_response, "The answer is 42.")
        self.assertEqual(trace.thinking_parts, ["I should calculate."])
        self.assertEqual((trace.input_tokens, trace.output_tokens), (100, 25))
        self.assertEqual(trace.turns, 1)
        self.assertEqual(trace.tool_trace[0]["name"], "bash")
        self.assertEqual(
            trace.tool_trace[0]["arguments"], {"command": "python3 solve.py"}
        )
        self.assertFalse(trace.tool_trace[0]["is_error"])
        self.assertEqual(trace.tool_trace[0]["output"], "42\n")

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


class HarnessPairingTests(unittest.TestCase):
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
                {"task_id": "a", "score": 0.0, "passed": False},
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
        self.assertEqual(pi["harness_delta_pp"], 50.0)
        self.assertEqual(pi["harness_gains"], 1)
        self.assertEqual(pi["harness_regressions"], 0)


if __name__ == "__main__":
    unittest.main()
