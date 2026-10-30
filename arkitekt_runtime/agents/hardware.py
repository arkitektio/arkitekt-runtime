"""What the machine an agent runs on has, as it says on every ``Register``.

A runtime formula divides by these (``agent.cpu_count``), so they are what this process may
actually use: a container's limits count, not the host's. Whatever cannot be read stays
unknown, and the frontend then shows no estimate rather than a wrong one.
"""

import functools
import math
import os
import subprocess
from pathlib import Path

from arkitekt_runtime.messages import AgentGpu, AgentHardware

_CGROUP = Path("/sys/fs/cgroup")
_NVIDIA_SMI = ("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits")
_NVIDIA_SMI_TIMEOUT = 2.0
_MIB = 1024 * 1024


def _read_cgroup(name: str) -> str | None:
    try:
        return (_CGROUP / name).read_text().strip()
    except OSError:
        return None


def _cpu_count() -> int | None:
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # not on every platform
        cpus = os.cpu_count()
    limit = _read_cgroup("cpu.max")
    if cpus is None or limit is None:
        return cpus
    quota, _, period = limit.partition(" ")
    if not quota.isdigit() or not period.isdigit() or int(period) == 0:  # "max 100000": no limit
        return cpus
    return min(cpus, max(1, math.ceil(int(quota) / int(period))))


def _memory_mib() -> int | None:
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):  # not on every platform
        return None
    limit = _read_cgroup("memory.max")
    if limit is not None and limit.isdigit():  # "max": no limit
        total = min(total, int(limit))
    return total // _MIB


def _gpus() -> list[AgentGpu]:
    try:
        listed = subprocess.run(_NVIDIA_SMI, capture_output=True, check=True, text=True, timeout=_NVIDIA_SMI_TIMEOUT).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in listed.splitlines():
        name, _, memory = line.rpartition(",")
        if name:
            gpus.append(AgentGpu(name=name.strip(), memory_mib=int(memory) if memory.strip().isdigit() else None))
    return gpus


@functools.cache
def detect_hardware() -> AgentHardware:
    """This process's hardware. Blocking the first time (``nvidia-smi``), and asked once per process."""
    return AgentHardware(cpu_count=_cpu_count(), memory_mib=_memory_mib(), gpus=_gpus())
