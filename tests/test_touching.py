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
