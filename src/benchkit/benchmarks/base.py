"""Benchmark base protocol."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Task:
    id: str
    prompt: str
    metadata: dict = field(default_factory=dict)


class Benchmark(Protocol):
    name: str

    def load_tasks(self) -> list[Task]: ...
    def build_prompt(self, task: Task) -> str: ...
    def evaluate(self, task: Task, response: str) -> bool: ...
