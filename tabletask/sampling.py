"""Drawing a concrete task for an episode, and sampling a start for it."""

import xml.etree.ElementTree as ET
from importlib import import_module
from pathlib import Path

import numpy as np

from . import fetching
from .language.errors import SamplingError, UnsupportedError
from .language.format import load
from .language.modules import require_capabilities
from .language.schema import Task
from .language.transformations import bind, concrete, draw, multiply, transform_task

BACKENDS = ("mujoco",)
DRAWS = 12  # draws of the variables and counts per episode, each placed up to STARTS times


def instantiate(source, episode=0, bindings=None) -> Task:
    """The concrete task for an episode: variables drawn and counts expanded.

    Args:
        source: A task id, task file, JSON string, dict or `Task`.
        episode: The seed for the draw.
        bindings: Assets for abstract objects; their properties must match the object's.
    """
    return _instantiate(load(source), np.random.default_rng(episode), bindings)


def labels(task: Task) -> dict[str, str]:
    """What each object is called in an instruction.

    An asset gives its name in MJCF metadata, which the simulator ignores:
    `<custom><text name="label" data="tin of tuna" /></custom>`. Without one, an
    object is called by its name with spaces for underscores.
    """
    names = {}
    for name, obj in task.scene.objects.items():
        names[name] = name.replace("_", " ")
        if not obj.asset:
            continue
        try:
            path = fetching.resolve(obj.asset, task.root)
            label = ET.parse(path).getroot().find("custom/text[@name='label']")
        except (OSError, ET.ParseError):
            continue
        if label is not None and label.get("data"):
            names[name] = label.get("data")
    return names


def _instantiate(task: Task, rng, bindings=None) -> Task:
    result = bind(task, draw(task, rng), labels(task))
    if bindings:  # before copies are made, so a counted object's copies share its binding
        result = _attach(result, bindings)
    return concrete(multiply(result, rng))


def _attach(task: Task, bindings: dict) -> Task:
    """Give abstract objects the assets in `bindings`, after checking their properties."""
    for name, binding in bindings.items():
        if name not in task.scene.objects:
            raise ValueError(f"bindings name an unknown object {name!r}")
        if (
            not isinstance(binding, dict)
            or set(binding) - {"asset", "properties"}
            or not binding.get("asset")
        ):
            raise ValueError(f"{name}: binding requires asset and optional properties")
        required = task.scene.objects[name].properties
        if any(binding.get("properties", {}).get(k) != v for k, v in required.items()):
            raise ValueError(f"{name}: binding does not meet required properties {required}")

    def assign(record):
        for name, binding in bindings.items():
            record["scene"]["objects"][name]["asset"] = binding["asset"]
        return record

    return transform_task(task, assign)


def build(source, episode=0, scene=None, bindings=None, *, robot=None, backend="mujoco"):
    """Sample a valid start for an episode and return the simulator's runtime.

    Raises:
        SamplingError: if no valid start was found within the sampling budget.
        UnsupportedError: if the task needs something the backend does not provide.
    """
    if backend not in BACKENDS:
        raise UnsupportedError("simulator", [backend])
    simulator = import_module(f".{backend}", __package__)
    template = load(source)
    require_capabilities(template, simulator.CAPABILITIES, f"{backend} observations")

    rng = np.random.default_rng(episode)
    last_error = None
    for _ in range(DRAWS):
        task = _instantiate(template, rng, bindings)
        if scene is not None:
            task.scene.file = str(Path(scene).resolve())
        unbound = [name for name, obj in task.scene.objects.items() if obj.asset is None]
        if unbound:
            raise UnsupportedError("asset binding", unbound)
        try:
            return simulator.build(task, rng, robot=robot)
        except SamplingError as error:
            last_error = error
    error = SamplingError(f"{last_error} (the last of {DRAWS} draws of the task)")
    error.runtime = getattr(last_error, "runtime", None)  # the last draw, for inspection
    raise error from last_error
