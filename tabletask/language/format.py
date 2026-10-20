"""Reading task documents (the JSON format) and compiling them to tasks."""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import msgspec

from . import grammar, kernel, schema
from .modules import load_modules
from .schema import ROTATIONS, Task
from .transformations import inline
from .vocabulary import (
    BOUND,
    CONNECTIVES,
    JOINT_WORDS,
    OTHER,
    POSE_WORDS,
    QUANTIFIERS,
    TIME_WORDS,
    VOCABULARY,
    container_region,
)

VERSION = "0.1"  # the document format version this reads
DEFAULT_SCENE = "main"
DEFAULT_LIBRARY = "../assets"  # asset library, relative to the task file
SCENES = Path(__file__).parent.parent / "scenes"
# The shipped tasks: inside an installed package, or beside the package in a checkout.
TASKS = next(
    (
        p
        for p in (Path(__file__).parents[1] / "data/tasks", Path(__file__).parents[2] / "tasks")
        if p.is_dir()
    ),
    None,
)
# Words with special meaning that are not relations, terms or defined words.
KEYWORDS = {*CONNECTIVES, *TIME_WORDS, "then", "fixed"}
# An object named like `mug_2` defaults to the asset of its base name, `mug`.
NUMBERED_NAME = re.compile(r"^(.*?)_(\d+)$")


class Document(msgspec.Struct, forbid_unknown_fields=True):
    """A task file."""

    id: str
    instruction: str
    start: list
    goal: list | dict
    tabletask: str = VERSION
    scene: str = DEFAULT_SCENE
    library: str = DEFAULT_LIBRARY
    appearance: schema.Appearance | None = None
    assets: dict[str, str] = {}
    objects: dict[str, dict] = {}
    fixtures: list[str] = msgspec.field(default_factory=lambda: ["table"])
    always: list = []
    uses: dict[str, dict] = {}
    states: dict[str, dict] = {}
    variables: dict[str, schema.Draw] = {}
    time_limit: float | None = None


class ObjectDocument(msgspec.Struct, forbid_unknown_fields=True):
    """An entry in a task file's `objects`."""

    asset: str | list[str] | None = None
    tags: list[str] = []
    properties: dict[str, Any] = {}
    fixed: bool = False
    count: int | tuple[int, int] = 1
    scale: float | list[float | list[float]] = 1.0
    sample: schema.Frame | str | None = None
    regions: dict[str, schema.Region] = {}


class StageDocument(msgspec.Struct, forbid_unknown_fields=True):
    """A named stage in a task file's `goal`."""

    condition: list
    after: list[str] = []
    hold: float = 0.5
    maintain: bool = False
    held: bool = False  # the stage counts while the robot still holds the object


def tasks(tier: str | None = None) -> list[str]:
    """The ids of the tasks that ship with tabletask, optionally of one tier ("base", ...)."""
    if TASKS is None:
        return []
    return sorted(p.stem for p in TASKS.glob(f"{tier or '*'}/*.json"))


def task_file(task_id: str) -> Path | None:
    """The file of a shipped task, or None if no shipped task has that id."""
    if TASKS is None:
        return None
    found = sorted(TASKS.glob(f"*/{task_id}.json"))
    return found[0] if found else None


def load(source) -> Task:
    """Load and check a task without loading the simulator or any asset.

    Args:
        source: A task id from `tasks()`, a path to a task file, a JSON string, a dict,
            or a `Task` (returned unchanged).

    Raises:
        FileNotFoundError: if `source` is neither a file nor a shipped task id.
        ValueError: if the document is malformed.
    """
    if isinstance(source, Task):
        return source
    if isinstance(source, dict):
        return compile_document(source)
    text = str(source)
    if text.lstrip().startswith("{"):
        return compile_document(json.loads(text))
    path = Path(source)
    if not path.exists() and not path.suffix:
        path = task_file(text)
        if path is None:
            near = get_close_matches(text, tasks(), 3)
            raise FileNotFoundError(
                f"no task file or task id {text!r}" + (f"; did you mean {near}?" if near else "")
            )
    path = path.resolve()
    return compile_document(json.loads(path.read_bytes()), path)


def dumps(doc: dict) -> str:
    """A task document as JSON text, one condition per line, as task files are written."""
    lines = ["{"]
    for i, (key, value) in enumerate(doc.items()):
        comma = "," if i < len(doc) - 1 else ""
        if key in ("start", "goal", "always") and isinstance(value, list):
            lines.append(f'  "{key}": [')
            for j, cond in enumerate(value):
                lines.append(
                    "    " + json.dumps(cond, allow_nan=False) + ("," if j < len(value) - 1 else "")
                )
            lines.append(f"  ]{comma}")
        else:
            lines.append(f'  "{key}": {json.dumps(value, allow_nan=False)}{comma}')
    return "\n".join([*lines, "}"]) + "\n"


def compile_document(doc: dict, source: Path | None = None) -> Task:
    """Compile a task document to a `Task`, checking its syntax and meaning.

    `source` is the file the document came from; relative paths resolve against it.
    """
    source = Path(source) if source else None
    root = source.parent if source else Path.cwd()
    document = _read_document(doc)
    local, observations = load_modules(document.uses, root)
    if set(local) & set(document.states):
        raise ValueError("task states must not shadow imported words")
    local.update(document.states)
    words = _vocabulary(local)
    fixtures = set(document.fixtures)
    entries = _parse_entries(document, fixtures, words)
    parser = _ConditionParser(
        words,
        fixtures,
        set(entries) | fixtures,
        {name for name, entry in entries.items() if entry.count != 1},
    )
    objects = _parse_objects(document, entries)
    sampled = {name for name, entry in entries.items() if entry.sample is not None}
    start = _parse_start(document.start, objects, parser, sampled)
    stages = _parse_goal(document.goal, parser)
    always = {f"always {i + 1}": parser.parse(c) for i, c in enumerate(document.always)}
    record = {
        "id": document.id,
        "instruction": document.instruction,
        "scene": {
            "file": _scene(document.scene, root),
            "library": document.library,
            "objects": objects,
            "appearance": msgspec.to_builtins(document.appearance),
            "start": {"and": start},
        },
        "goal": {"stages": stages, "always": always},
        "states": local,
        "variables": msgspec.to_builtins(document.variables),
        "time_limit": document.time_limit,
        "observations": observations,
    }
    for definition in local.values():
        if isinstance(definition, dict) and isinstance(definition.get("is"), list):
            definition["is"] = parser.parse(definition["is"])
    # through JSON text, which rejects NaN and infinity anywhere in the record
    task = grammar.check_task(msgspec.json.decode(json.dumps(record, allow_nan=False), type=Task))
    task.source = source
    _check_meaning(task, parser)
    task.authored = copy.deepcopy(doc)
    return task


def _vocabulary(local: dict) -> set[str]:
    """Every word a task may use: the built-in ones, and those it imports or defines."""
    return (
        set(VOCABULARY)
        | KEYWORDS
        | set(QUANTIFIERS)
        | set(kernel.RELATIONS)
        | set(kernel.TERMS)
        | set(kernel.ARITHMETIC)
        | set(local)
    )


def _scene(scene: str, root: Path) -> str:
    """The path of a scene: a bare name is a shipped scene, anything else a file."""
    file = Path(scene)
    if not file.suffix:
        return str(SCENES / f"{file}.xml")
    return str(file if file.is_absolute() else root / file)


def _check_meaning(task, parser) -> None:
    """Check that every name is declared and every condition is true or false."""
    expanded = inline(task)
    for name, _ in grammar.names(expanded):
        if name.startswith("#") or name in (BOUND, OTHER) or "{" in name:
            continue
        parser.reference(name)

    for stage in expanded.goal.stages.values():
        _require_boolean(stage.condition)
    _require_boolean(expanded.scene.start)
    for condition in expanded.goal.always.values():
        _require_boolean(condition)


def _read_document(doc: dict) -> Document:
    """Check a document's keys and types."""
    allowed = set(Document.__struct_fields__)
    if not isinstance(doc, dict):
        raise ValueError("a task document must be an object")
    if set(doc) - allowed:
        raise ValueError(f"{doc.get('id')}: unknown keys {sorted(set(doc) - allowed)}")
    try:
        document = msgspec.convert(doc, type=Document)
    except (msgspec.ValidationError, TypeError) as error:
        raise ValueError(str(error)) from error
    if document.tabletask not in ("0.0", VERSION):
        raise ValueError(
            f"this reads tabletask 0.0 and {VERSION}; the file says {document.tabletask!r}"
        )
    if not document.id or not document.instruction:
        raise ValueError("id and instruction must not be empty")
    if document.time_limit is not None and (
        not math.isfinite(document.time_limit) or document.time_limit <= 0
    ):
        raise ValueError("time_limit must be finite and positive")
    return document


@dataclass
class _ConditionParser:
    """Compiles conditions from list form, `[word, arg, ...]`, to `{word: [args]}`."""

    words: set[str]  # every word a condition may use
    fixtures: set[str]  # scene fixtures, such as the table
    known: set[str]  # every object and fixture name
    counted: set[str]  # objects with a count, whose copies are named mug1, mug2, ...

    def reference(self, value):
        """Check that a name refers to a known object or fixture; return it unchanged."""
        if not isinstance(value, str) or value.startswith("#") or "{" in value:
            return value
        head = value.split("/", 1)[0]
        if head in self.counted:
            raise ValueError(
                f"{head!r} has a count, so its copies are {head}1, {head}2, ...; "
                f"refer to all of them as #{head}"
            )
        copy = any(head.startswith(c) and head[len(c) :].isdigit() for c in self.counted)
        if value not in self.known and head not in self.known and not copy:
            raise ValueError(f"unknown object or fixture {value!r}")
        return value

    def parse(self, cond):
        """Compile one condition and the conditions nested in it."""
        if not isinstance(cond, list) or not cond or not isinstance(cond[0], str):
            raise ValueError(f"a condition is [word, ...]; got {cond!r}")
        word, args = cond[0], cond[1:]
        if word not in self.words:
            near = get_close_matches(word, sorted(self.words), 3)
            raise ValueError(
                f"unknown word {word!r}" + (f"; did you mean {', '.join(near)}?" if near else "")
            )
        if word in ("then", "fixed"):
            where = "as the whole goal" if word == "then" else "in start"
            raise ValueError(f"`{word}` is only permitted {where}")

        out = [self.argument(a) for a in args]
        if word == "oriented" and len(out) > 1 and isinstance(out[1], str):
            if out[1] not in ROTATIONS:
                raise ValueError(f"unknown rotation {out[1]!r}")
            out[1] = list(ROTATIONS[out[1]])
        if word == "joint" and len(out) == 2:
            out.append(out[1])
        held = container_region(word, out)
        if held != out:
            out = held
        else:  # a bare fixture name means its top: `on x table` is `on x table/top`
            out = [a + "/top" if isinstance(a, str) and a in self.fixtures else a for a in out]
        return {word: out}

    def argument(self, value):
        """Compile an argument: a nested condition, a list, or a literal."""
        if isinstance(value, list):
            if value and isinstance(value[0], str) and value[0] in self.words:
                return self.parse(value)
            return [self.argument(v) for v in value]
        if isinstance(value, dict):
            raise ValueError("expressions use arrays, not objects")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("condition numbers must be finite")
        return value


def _parse_entries(
    document: Document, fixtures: set[str], words: set[str]
) -> dict[str, ObjectDocument]:
    """Every object: those in `objects`, and any the start names that are not there."""
    if len(fixtures) != len(document.fixtures):
        raise ValueError("fixtures must be unique")
    entries = {
        name: msgspec.convert(value, type=ObjectDocument)
        for name, value in document.objects.items()
    }

    counted = {name for name, entry in entries.items() if entry.count != 1}
    for cond in document.start:
        if (
            isinstance(cond, list)
            and len(cond) >= 2
            and cond[0] in ("inside", "on", "at", "fixed", *POSE_WORDS, *JOINT_WORDS)
            and isinstance(cond[1], str)
        ):
            name = cond[1].split("/", 1)[0]
            copy = any(name.startswith(c) and name[len(c) :].isdigit() for c in counted)
            if name not in fixtures and name not in entries and not copy:
                m = NUMBERED_NAME.match(name)
                entries[name] = ObjectDocument(
                    asset=document.assets.get(name, m.group(1) if m else name)
                )
    for name in entries:
        if any(name.startswith(c) and name[len(c) :].isdigit() for c in counted):
            raise ValueError(f"{name!r} would collide with a copy of a counted object")
        if name in fixtures or name in words or "/" in name or "#" in name or not name:
            raise ValueError(f"invalid or reserved object name {name!r}")
    if set(document.assets) - set(entries):
        raise ValueError("assets names objects the task does not have")
    return entries


def _parse_objects(document: Document, entries: dict[str, ObjectDocument]) -> dict:
    """Each object as a compiled record, placed on the table unless sampled elsewhere."""
    objects = {}
    for name, entry in entries.items():
        count = entry.count
        bounds = (count, count) if isinstance(count, int) else count
        if bounds[0] < 1 or bounds[1] < bounds[0]:
            raise ValueError(f"{name}: count must be positive and ordered")
        _check_scale(name, entry.scale)
        asset = document.assets.get(name, entry.asset)
        if isinstance(asset, list):
            if len(asset) < bounds[1]:
                raise ValueError(f"{name}: {len(asset)} assets cannot give {bounds[1]} copies")
            asset = [_asset_path(document, a) for a in asset]
        elif asset:
            asset = _asset_path(document, asset)
        sample = msgspec.to_builtins(entry.sample) if entry.sample is not None else "table/top"
        objects[name] = {
            "sample": {"rel": sample} if isinstance(sample, str) else sample,
            "asset": asset,
            "tags": entry.tags,
            "properties": entry.properties,
            "fixed": entry.fixed,
            "count": entry.count,
            "scale": entry.scale,
            "regions": msgspec.to_builtins(entry.regions),
        }
    return objects


def _asset_path(document: Document, asset: str) -> str:
    """An asset name resolved to its file in the task's library; a path stays as written."""
    return asset if Path(asset).suffix else f"{document.library}/objects/{asset}.xml"


def _check_scale(name: str, scale) -> None:
    """Reject a scale that is not a factor, a `[low, high]` range, or one of either per axis."""
    if not isinstance(scale, list):
        ranges = [[scale, scale]]
    elif len(scale) == 2 and not any(isinstance(entry, list) for entry in scale):
        ranges = [scale]
    elif len(scale) == 3:
        ranges = [entry if isinstance(entry, list) else [entry, entry] for entry in scale]
    else:
        ranges = []
    if not ranges or any(len(r) != 2 or not 0 < r[0] <= r[1] for r in ranges):
        raise ValueError(
            f"{name}: scale is a positive factor, a [low, high] range, or one of either per axis"
        )


def _parse_start(
    conditions: list, objects: dict, parser: _ConditionParser, sampled: set[str]
) -> list:
    """Compile the start conditions, and set the placement and joint state they imply.

    `sampled` are the objects whose `sample` the task file writes; a start placement
    must agree with it.
    """
    placed = set()
    start = []
    for cond in conditions:
        if not isinstance(cond, list) or not cond:
            raise ValueError("start must contain conditions")
        word, args = cond[0], cond[1:]
        if word == "fixed":
            if len(args) != 1 or args[0] not in objects:
                raise ValueError("fixed takes one object")
            objects[args[0]]["fixed"] = True
            continue
        compiled = parser.parse(cond)
        who = args[0] if args and isinstance(args[0], str) else None
        placing = word in ("on", "inside", "at") and len(args) == 2
        if who in objects and objects[who]["count"] != 1:
            # the copies take a placement or pose as their sample; nothing else can apply,
            # and `at` would put every copy within 2 cm of one point
            turning = word in POSE_WORDS and len(args) == (2 if word == "oriented" else 1)
            if not ((placing and word != "at") or turning):
                raise ValueError(
                    f"{who} has a count: its start can place it (on or inside, with no "
                    "tolerance) or turn it (with no tolerance)"
                )
        else:
            start.append(compiled)

        if (
            placing
            and who not in objects
            and any(str(who).startswith(c) and str(who)[len(c) :].isdigit() for c in parser.counted)
        ):
            raise ValueError(
                f"{who} is a copy of a counted object; place the copies with the object's sample"
            )
        if placing and who in objects:
            where = parser.reference(args[1])
            if not isinstance(where, str):
                raise ValueError(f"{word}: placement target must be a name")
            if where == who:
                raise ValueError(f"{who}: cannot be placed on itself")
            if who in placed:
                raise ValueError(f"{who!r} is placed twice in the start")
            placed.add(who)
            rel = where if "/" in where else f"{where}/{'inside' if word == 'inside' else 'top'}"
            hint = objects[who]["sample"]
            # `on x table` with sample `table/left` agree; an explicit region must match,
            # and so must `at`, which measures from the centre of the region it names
            bare = "/" not in where and word != "at"
            agrees = hint["rel"] == rel or (bare and hint["rel"].split("/")[0] == where)
            if who in sampled and not agrees:
                raise ValueError(
                    f"{who}: its sample is relative to {hint['rel']!r}, but the start puts "
                    f"it {word} {where!r}; remove the sample or make them agree"
                )
            # a written sample says where on that object; otherwise replace the default
            # placement, keeping a rotation set earlier
            if who not in sampled and hint["rel"] != rel:
                kept = {k: hint[k] for k in ("rotation", "jitter") if k in hint}
                objects[who]["sample"] = {"rel": rel, **kept}
            if word == "at":
                objects[who]["sample"].setdefault("jitter", {}).setdefault("xy", 0.02)
        elif word in POSE_WORDS and who in objects:
            rotation = (
                args[1]
                if word == "oriented" and len(args) > 1
                else {"lying": "on_side"}.get(word, word)
            )
            frame = objects[who]["sample"]
            frame["rotation"] = rotation
            if word == "oriented":
                frame.setdefault("jitter", {})["yaw"] = 0.0
        elif word in JOINT_WORDS and who and "/" in who:
            host, joint = who.split("/", 1)
            if host not in objects:
                raise ValueError(f"unknown joint owner {host}")
            if objects[host]["count"] != 1:
                raise ValueError(f"{host} has a count; the start cannot set its copies' joints")
            value = word if word in ("open", "closed") else args[1] if len(args) == 2 else args[1:]
            objects[host].setdefault("joints", {})[joint] = value

    return start


def _condition_label(c):
    """A condition as words, used to name the stage it makes."""
    return " ".join(_condition_label(v) if isinstance(v, list) else str(v) for v in c)


def _parse_goal(goal, parser: _ConditionParser) -> dict:
    """Compile the goal to named stages: one per `then` step, or as written."""
    stages = {}
    if isinstance(goal, dict):
        if set(goal) != {"stages"} or not isinstance(goal["stages"], dict):
            raise ValueError(
                "an explicit goal is {stages: {name: {condition, after, hold, maintain, held}}}"
            )
        for name, spec in goal["stages"].items():
            stage = msgspec.convert(spec, type=StageDocument)
            if not name or not math.isfinite(stage.hold) or stage.hold < 0:
                raise ValueError("stage name must be nonempty and hold finite and nonnegative")
            stages[name] = {
                "condition": parser.parse(stage.condition),
                "after": stage.after,
                "hold": stage.hold,
                "maintain": stage.maintain,
                "held": stage.held,
            }
    else:
        if goal and isinstance(goal[0], str):
            goal = [goal]
        sequential = (
            len(goal) == 1 and isinstance(goal[0], list) and bool(goal[0]) and goal[0][0] == "then"
        )
        parts = goal[0][1:] if sequential else [goal]
        previous = None
        for part in parts:
            conditions = part if part and isinstance(part[0], list) else [part]
            if not conditions or not conditions[0]:
                raise ValueError("a task needs a goal")
            condition = (
                parser.parse(conditions[0])
                if len(conditions) == 1
                else {"and": [parser.parse(c) for c in conditions]}
            )

            base = " and ".join(_condition_label(c) for c in conditions)
            name, i = base, 2
            while name in stages:
                name, i = f"{base} ({i})", i + 1
            stages[name] = {"condition": condition, "after": [previous] if previous else []}
            previous = name
    if not stages:
        raise ValueError("a task needs a goal")
    return stages


def _require_boolean(node):
    """Raise ValueError if a condition is a number rather than true or false."""
    name, args = next(iter(node.items()))
    if name in CONNECTIVES:
        for child in args:
            _require_boolean(child)
    elif name in TIME_WORDS:
        _require_boolean(args[0])
    elif name == "each":
        raise ValueError("each makes a list of conditions; use it inside tally")
    elif name not in kernel.RELATIONS and name not in QUANTIFIERS:
        raise ValueError(f"{name}: a condition must produce a boolean, not a number")


def json_schema() -> dict:
    """The JSON Schema of a task file. It checks structure; `load` also checks meaning."""
    result = msgspec.json.schema(Document)
    for struct in (ObjectDocument, StageDocument):
        result["$defs"].update(msgspec.json.schema(struct)["$defs"])
    properties = result["$defs"]["Document"]["properties"]
    properties["objects"] = {
        "type": "object",
        "additionalProperties": {"$ref": "#/$defs/ObjectDocument"},
    }
    properties["goal"] = {
        "anyOf": [
            {"type": "array"},
            {
                "type": "object",
                "required": ["stages"],
                "additionalProperties": False,
                "properties": {
                    "stages": {
                        "type": "object",
                        "minProperties": 1,
                        "additionalProperties": {"$ref": "#/$defs/StageDocument"},
                    }
                },
            },
        ]
    }
    return result
