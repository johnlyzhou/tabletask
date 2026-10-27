"""Task sampling and simulator dispatch for episode starts."""

import functools
import xml.etree.ElementTree as ET
from copy import deepcopy
from importlib import import_module
from pathlib import Path

import numpy as np

from . import fetching
from .episodes import ResetState
from .generation import realize_task
from .language.errors import SamplingError, UnsupportedError
from .language.format import load
from .language.modules import require_capabilities
from .language.schema import Task
from .language.transformations import bind, concrete, draw, multiply, transform_task
from .language.vocabulary import PLACEHOLDER

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
    """What each object is called: its asset's first label.

    An asset gives its names in MJCF metadata, which the simulator ignores:
    `<custom><text name="label" data="tin of tuna|tuna can|can" /></custom>`, the first
    being its main name and the rest other ways to say it. Without one, an object is
    called by its name with spaces for underscores.
    """
    names = {}
    for name, obj in task.scene.objects.items():
        said = _asset_labels(task, obj.asset)
        names[name] = said[0] if said else name.replace("_", " ")
    return names


def _asset_text(task: Task, asset, key: str) -> str | None:
    """A `<custom><text>` entry an asset file declares, such as its label or category."""
    return _entry(task.root, asset, key)


def _entry(root: Path, asset, key: str) -> str | None:
    """A `<custom><text>` entry of the asset file `asset` names, found from `root`."""
    if not isinstance(asset, str):
        return None
    try:
        path = fetching.resolve(asset, root)
        text = ET.parse(path).getroot().find(f"custom/text[@name='{key}']")
    except (OSError, ET.ParseError):
        return None
    return text.get("data") if text is not None and text.get("data") else None


def _asset_labels(task: Task, asset) -> list[str]:
    """The names an asset file declares, main name first; empty if it names none."""
    text = _asset_text(task, asset, "label")
    return [name.strip() for name in text.split("|") if name.strip()] if text else []


@functools.lru_cache(maxsize=None)
def _groups(root, pool: tuple, key: str) -> list[list[str]]:
    """A pool's assets grouped by the entry `key` they declare; one without it is its own group.

    Cached, so a pool's asset files are read once per process rather than on every draw.
    """
    split = {}
    for asset in pool:
        split.setdefault(_entry(root, asset, key) or asset, []).append(asset)
    return list(split.values())


def _apart(task: Task) -> bool:
    """Whether no two objects share the asset entry the task keeps `separate` (its category)."""
    seen = [_asset_text(task, obj.asset, task.separate) for obj in task.scene.objects.values()]
    seen = [value for value in seen if value]
    return len(seen) == len(set(seen))


def _speak(task: Task, rng=None) -> Task:
    """Name objects in the instruction: `{mug1}` becomes one of its asset's labels.

    With `rng`, each object is called by a random one of its labels that no other
    object in the scene shares, so the instruction stays unambiguous; when every label is
    shared, or without `rng`, it is called by its main name.
    """
    if not PLACEHOLDER.search(task.instruction):
        return task
    said = {
        name: _asset_labels(task, obj.asset) or [Path(obj.asset).stem.replace("_", " ")]
        for name, obj in task.scene.objects.items()
        if obj.asset
    }
    names = {}
    for name, options in said.items():
        taken = {label for other, labels in said.items() if other != name for label in labels}
        free = [label for label in options if label not in taken]
        names[name] = str(rng.choice(free)) if rng is not None and free else options[0]
    return transform_task(
        task,
        lambda record: {
            **record,
            "__object_labels__": names,
            "instruction": PLACEHOLDER.sub(
                lambda m: names.get(m.group(1), m.group(0)), record["instruction"]
            ),
        },
    )


def _instantiate(task: Task, rng, bindings=None) -> Task:
    # words draw from their own stream, so the phrasing and names chosen never change where
    # things are placed
    words = rng.spawn(1)[0]
    if task.phrasings:
        said = [task.instruction, *task.phrasings]
        task = transform_task(
            task,
            lambda record: {
                **record,
                "instruction": said[int(words.integers(len(said)))],
                "phrasings": [],
            },
        )
    values = draw(task, rng)
    result = bind(task, values, labels(task))
    result.drawn = values
    if bindings:  # before copies are made, so a counted object's copies share its binding
        result = _attach(result, bindings)
    # a task that keeps a category `separate` draws each pool's copies from distinct
    # categories, every category equally likely
    groups = (lambda pool: _groups(result.root, tuple(pool), task.separate)) if task.separate else None
    copies = multiply(result, rng, groups)
    # copies of different objects may still share one (a bowl item and a bowl container),
    # so such a draw is made again
    for _ in range(DRAWS * 4):
        if not task.separate or _apart(copies):
            break
        copies = multiply(result, rng, groups)
    else:
        raise SamplingError(f"{task.id}: could not draw objects with no two sharing a {task.separate}")
    return _speak(concrete(realize_task(copies, rng)), words)


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


def capture(runtime, template, seed, *, backend="mujoco"):
    """Capture a detached initial state for replay.

    Args:
        runtime: Live simulator runtime.
        template: Source task defining observation padding.
        seed: Sampling seed.
        backend: Simulator name.
    """
    simulator = import_module(f".{backend}", __package__)
    return ResetState(
        seed=seed,
        task=deepcopy(runtime.task),
        template=deepcopy(template),
        backend=backend,
        backend_state=simulator.capture(runtime),
    )


def restore(state, task=None):
    """Restore a captured start with an optional replacement objective.

    Args:
        state: Captured ResetState.
        task: Task source retaining the captured scene, or None to retain its objective.
    """
    task = deepcopy(state.task if task is None else load(task))
    if task.scene != state.task.scene or task.root != state.task.root or task.variables:
        raise ValueError("a replay task must retain the captured concrete scene and asset root")
    simulator = import_module(f".{state.backend}", __package__)
    return simulator.restore(task, state.backend_state)


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
