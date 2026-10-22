"""Procedural assets: generator scripts that write an asset's MJCF from drawn parameters.

A generator is a Python file in the library's `generators/` folder (it may import the
modules beside it) with two names:
`PARAMS`, each parameter's fixed value, `[low, high]` range or list of choices, and
`build(params) -> str`, the MJCF of one asset (one top-level body) for one set of drawn
values. A task's object
names one with `{"generate": name, "params": {...}}`; its params narrow `PARAMS`. Each
episode draws the values and the asset is written to a cache file named by its
content, so everything after sampling sees an ordinary asset file.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import xml.etree.ElementTree as ET
from functools import cache
from pathlib import Path

import numpy as np

from . import fetching
from .language.schema import Generate, Task
from .language.transformations import transform_task

CACHE = Path(
    os.environ.get("TABLETASK_GENERATED", Path.home() / ".cache" / "tabletask" / "generated")
)


@cache
def generator(path: Path):
    """The generator module at `path`, loaded once."""
    spec = importlib.util.spec_from_file_location(f"tabletask_generator_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))  # a generator may import the modules beside it
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    if not isinstance(getattr(module, "PARAMS", None), dict) or not callable(
        getattr(module, "build", None)
    ):
        raise ValueError(f"{path.name} must define PARAMS and build(params)")
    return module


def draw_params(defaults: dict, given: dict, rng=None) -> dict:
    """One value per parameter: given values narrow the defaults; without `rng`, the middle.

    Ranges are drawn uniformly and choices evenly, in the order of `defaults`.
    """
    unknown = set(given) - set(defaults)
    if unknown:
        raise ValueError(
            f"unknown generator parameters {sorted(unknown)}; it takes {sorted(defaults)}"
        )
    values = {}
    for name, default in defaults.items():
        spec = given.get(name, default)
        if _is_range(spec):
            low, high = float(spec[0]), float(spec[1])
            values[name] = (low + high) / 2 if rng is None else float(rng.uniform(low, high))
        elif isinstance(spec, list):
            values[name] = spec[0] if rng is None else spec[int(rng.integers(len(spec)))]
        else:
            values[name] = spec
    return values


def _is_range(spec) -> bool:
    return (
        isinstance(spec, list)
        and len(spec) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in spec)
    )


def realize(asset: Generate, root: Path, rng=None) -> str:
    """Draw a generator's parameters, write its asset to the cache, and return the file."""
    module = generator(fetching.resolve(asset.generate, root))
    xml = module.build(draw_params(module.PARAMS, asset.params, rng))
    digest = hashlib.sha1(xml.encode()).hexdigest()[:16]
    path = CACHE / f"{Path(asset.generate).stem}-{digest}.xml"
    if not path.exists():
        root = ET.fromstring(xml)  # an asset's model and its one body are named after its file
        bodies = root.findall("worldbody/body")
        if len(bodies) != 1:
            raise ValueError(f"{Path(asset.generate).name} must build one top-level body")
        old = bodies[0].get("name")
        root.set("model", path.stem)
        for element in root.iter():
            for key in ("name", "body", "body1", "body2"):
                if element.get(key) == old and (key != "name" or element is bodies[0]):
                    element.set(key, path.stem)
        CACHE.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(f".{os.getpid()}.part")
        part.write_text(ET.tostring(root, encoding="unicode"))
        part.replace(path)
    return str(path)


def realize_task(task: Task, rng) -> Task:
    """The task with each generated object's asset drawn and written to a file.

    The draws come from their own stream, seeded by one draw made only when the task
    has a generated object, so tasks without one sample as before.
    """
    generated = [
        name for name, obj in task.scene.objects.items() if isinstance(obj.asset, Generate)
    ]
    if not generated:
        return task
    stream = np.random.default_rng(int(rng.integers(2**63))) if rng is not None else None
    files = {name: realize(task.scene.objects[name].asset, task.root, stream) for name in generated}

    def write(record):
        for name, file in files.items():
            record["scene"]["objects"][name]["asset"] = file
        return record

    return transform_task(task, write)
