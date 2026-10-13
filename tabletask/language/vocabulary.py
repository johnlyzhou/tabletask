"""The words conditions are written in, beyond the kernel's relations and terms."""

import re
from pathlib import Path

import msgspec

# Words defined in terms of other words, such as `on` and `inside`.
VOCABULARY = msgspec.json.decode((Path(__file__).parent / "vocabulary.json").read_bytes())
# Logic: the connectives, words judged over the episode's history, and all time words.
CONNECTIVES = ("and", "or", "not")
TEMPORAL = ("once", "times")
TIME_WORDS = (*TEMPORAL, "initially")
# Each quantifier and the connective its expansion joins the bodies with.
QUANTIFIERS = {"every": "and", "some": "or", "no": "none", "every_pair": "pairs", "each": "list"}
# Start words that set an object's pose, and those that set a joint.
POSE_WORDS = ("upright", "oriented", "lying", "upside_down")
JOINT_WORDS = ("open", "closed", "joint")
# Default names a quantifier binds its members to.
BOUND, OTHER = "it", "other"
# A variable in a task's text, written `{name}`.
PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z_0-9]*)\}")


def container_region(word: str, args: list) -> list:
    """Point a bare container at its inside: `inside(x, bowl)` means `bowl/inside`."""
    if (
        word in ("inside", "count_in")
        and len(args) > 1
        and isinstance(args[1], str)
        and "/" not in args[1]
    ):
        args = [args[0], args[1] + "/inside", *args[2:]]
    return args
