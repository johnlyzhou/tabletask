"""An asset's several labels: instructions call an object by a random one no other object shares."""

import re
from pathlib import Path

import numpy as np

import tabletask
from tabletask.language.format import load
from tabletask.sampling import _instantiate, labels

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "objects"


def labelled(folder: Path, asset: str, label: str) -> str:
    """A copy of a library asset whose label is `label`; returns its path."""
    text = (ASSETS / f"{asset}.xml").read_text()
    out = folder / f"{asset}.xml"
    out.write_text(re.sub(r'<text name="label" data="[^"]*"', f'<text name="label" data="{label}"', text))
    return str(out)


def task(red: str, blue: str) -> dict:
    sample = {"rel": "table/top"}
    return {
        "id": "labels",
        "instruction": "Pick up the {red}, not the {blue}.",
        "objects": {"red": {"asset": red, "sample": sample}, "blue": {"asset": blue, "sample": sample}},
        "start": [["on", "red", "table"], ["on", "blue", "table"]],
        "goal": [["lifted", "red", 0.05, "table"]],
    }


def test_random_unshared_label(tmp_path):
    red = labelled(tmp_path, "red_block", "red block|block|red cube")
    blue = labelled(tmp_path, "blue_block", "blue block|block")
    said = {tabletask.instantiate(task(red, blue), episode).instruction for episode in range(40)}
    reds = {re.match(r"Pick up the (.*), not", s).group(1) for s in said}
    assert reds == {"red block", "red cube"}  # never "block": the blue one is a block too
    assert all(s.endswith("not the blue block.") for s in said)  # its only unshared label
    # the same episode always says the same thing
    assert tabletask.instantiate(task(red, blue), 7).instruction == tabletask.instantiate(task(red, blue), 7).instruction


def test_labels_do_not_move_things(tmp_path):
    """Naming draws from its own stream: the draws left for placing objects are the same."""
    blue = labelled(tmp_path, "blue_block", "blue block")
    (tmp_path / "plain").mkdir()
    plain = task(labelled(tmp_path / "plain", "red_block", "red block"), blue)
    varied = task(labelled(tmp_path, "red_block", "red block|block|red cube"), blue)
    rngs = [np.random.default_rng(3), np.random.default_rng(3)]
    tasks = [_instantiate(load(t), rng) for t, rng in zip((plain, varied), rngs)]
    assert rngs[0].random() == rngs[1].random()
    assert labels(tasks[1])["red"] == "red block"  # the main name is the first


def test_phrasings(tmp_path):
    """Each episode says one of the instruction's phrasings, without moving anything."""
    blue = labelled(tmp_path, "blue_block", "blue block")
    plain = task(labelled(tmp_path, "red_block", "red block"), blue)
    phrased = {**plain, "phrasings": ["Lift the {red}, leave the {blue}.", "Grab the {red}, not the {blue}."]}
    said = {tabletask.instantiate(phrased, episode).instruction for episode in range(40)}
    assert said == {"Pick up the red block, not the blue block.", "Lift the red block, leave the blue block.",
                    "Grab the red block, not the blue block."}
    rngs = [np.random.default_rng(5), np.random.default_rng(5)]
    for t, rng in zip((plain, phrased), rngs):
        _instantiate(load(t), rng)
    assert rngs[0].random() == rngs[1].random()
