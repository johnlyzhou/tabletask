"""`only_touching`: an object touches one thing and nothing else, the robot aside."""

from pathlib import Path

import tabletask
from tabletask.inspection import put
from tabletask.language.evaluation import evaluate

ROOT = Path(__file__).resolve().parent.parent


def test_only_touching():
    sample = {"rel": "table/top", "offset": [-0.07, 0.0, 0.0]}
    task = {
        "id": "touching",
        "instruction": "Put the block in the bowl.",
        "objects": {"bowl": {"asset": str(ROOT / "assets/objects/bowl.xml"), "sample": sample},
                    "block": {"asset": str(ROOT / "assets/objects/red_block.xml"), "sample": {"rel": "table/top"}}},
        "start": [["on", "bowl", "table"], ["on", "block", "table"]],
        "goal": [["inside", "block", "bowl"]],
    }
    env = tabletask.make(task, robot=None)
    env.reset(0)
    env.step(seconds=1.0)
    assert not evaluate({"only_touching": ["block", "bowl"]}, env.runtime)  # on the table
    assert env.runtime.touched("block") == {"table"}
    put(env.runtime, "block", "bowl")
    env.step(seconds=1.5)
    assert evaluate({"only_touching": ["block", "bowl"]}, env.runtime)
    assert env.runtime.touched("bowl") == {"table", "block"}


def test_side_of():
    """Sides as the robot sees them: left is +y, front is nearer the robot (-x)."""
    block = lambda name: {"asset": str(ROOT / f"assets/objects/{name}.xml"), "sample": {"rel": "table/top"}}
    task = {"id": "sides", "instruction": "Move the red cube.", "objects": {"red": block("red_cube"), "blue": block("blue_cube")},
            "start": [["on", "red", "table"], ["on", "blue", "table"]], "goal": [["side_of", "red", "blue", "left"]]}
    env = tabletask.make(task, robot=None)
    env.reset(0)
    rt = env.runtime
    blue, _ = rt.pose("blue")
    for side, step in {"left": (0, 0.08), "right": (0, -0.08), "front": (-0.08, 0), "back": (0.08, 0)}.items():
        _, turn = rt.pose("red")
        rt.set_pose("red", blue + [step[0], step[1], 0.0], turn)
        assert [s for s in ("left", "right", "front", "back") if evaluate({"side_of": ["red", "blue", s]}, rt)] == [side]
