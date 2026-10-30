"""An agent says what its machine has: what its process may use, and unknown when it cannot tell."""

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from arkitekt_runtime.agents import hardware
from arkitekt_runtime.messages import AgentGpu, AgentHardware, Register


@pytest.fixture(autouse=True)
def fresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setattr(hardware, "_CGROUP", tmp_path)
    hardware.detect_hardware.cache_clear()
    yield
    hardware.detect_hardware.cache_clear()


def _affinity(monkeypatch: pytest.MonkeyPatch, cpus: int) -> None:
    monkeypatch.setattr(hardware.os, "sched_getaffinity", lambda _: set(range(cpus)), raising=False)


def test_the_cpu_count_is_the_container_limit_when_there_is_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _affinity(monkeypatch, 32)
    (tmp_path / "cpu.max").write_text("250000 100000\n")

    assert hardware._cpu_count() == 3


def test_no_cpu_limit_leaves_the_affinity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _affinity(monkeypatch, 8)
    (tmp_path / "cpu.max").write_text("max 100000\n")

    assert hardware._cpu_count() == 8


def test_gpus_are_listed_by_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    listed = "NVIDIA GeForce RTX 4090, 24564\nA, Strange, Name, [N/A]\n"
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=listed))

    assert hardware._gpus() == [AgentGpu(name="NVIDIA GeForce RTX 4090", memory_mib=24564), AgentGpu(name="A, Strange, Name", memory_mib=None)]


def test_no_working_nvidia_smi_is_no_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(*_: object, **__: object) -> None:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(hardware.subprocess, "run", run)

    assert hardware._gpus() == []


def test_register_carries_it_on_the_wire() -> None:
    message = Register(token="t", hardware=AgentHardware(cpu_count=4, memory_mib=2048, gpus=[AgentGpu(name="G", memory_mib=8)]))

    assert '"hardware":{"cpu_count":4,"memory_mib":2048,"gpus":[{"name":"G","memory_mib":8}]}' in message.model_dump_json()
