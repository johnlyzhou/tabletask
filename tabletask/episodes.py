"""Captured episode starts for replay without resampling the physical scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .language.schema import Task


@dataclass(frozen=True)
class ResetState:
    """A replayable start returned in info["reset_state"] when capture is requested.

    Args:
        seed: Sampling seed.
        task: Concrete sampled task.
        template: Source task defining observation padding.
        backend: Simulator name.
        backend_state: Opaque scene and physics data owned by the simulator.
    """

    seed: int
    task: Task
    template: Task
    backend: str
    backend_state: Any
