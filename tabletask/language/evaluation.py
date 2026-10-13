"""Evaluating compiled conditions against a scene state."""

from __future__ import annotations

import json
from typing import Any

from .kernel import ARITHMETIC, RELATIONS, TERMS
from .state import State
from .vocabulary import CONNECTIVES, TEMPORAL, TIME_WORDS

LOGIC = (*CONNECTIVES, *TIME_WORDS)


def unpack_expression(tree) -> tuple[str, list]:
    """A condition as the word it is and the arguments it carries."""
    word = next(iter(tree))
    args = tree[word]
    return word, args if isinstance(args, list) else [args]


def evaluate_term(node: Any, s: State, history: dict | None = None):
    """A term call, a condition, or a plain literal, reduced to its value."""
    if isinstance(node, list):
        return [evaluate_term(v, s, history) for v in node]
    if not isinstance(node, dict):
        return node
    name, args = unpack_expression(node)
    if name in RELATIONS or name in LOGIC:
        return evaluate(node, s, history)
    if name in ARITHMETIC:
        left, right = (evaluate_term(a, s, history) for a in args)
        return ARITHMETIC[name](left, right)
    return TERMS[name](s, *[evaluate_term(a, s, history) for a in args])


def evaluate(tree: dict[str, Any], s: State, history: dict | None = None) -> bool:
    """Evaluate a condition; temporal operators use history when supplied."""
    name, args = unpack_expression(tree)

    if name == "initially":
        key = json.dumps(["initially", args[0]], sort_keys=True)
        if history is not None and key in history:
            return history[key]
        with s.rewind():
            answer = evaluate(args[0], s, None)
        if history is not None:
            history[key] = answer
        return answer

    if name in TEMPORAL:
        wanted = 1 if name == "once" else int(args[1])
        happening = evaluate(args[0], s, history)
        if history is None:
            return happening
        key = json.dumps([name, args[0]], sort_keys=True)
        seen = history.setdefault(key, {"count": 0, "was": None})
        # `once` counts a condition already true at the first check; `times` counts only
        # false-to-true transitions, so the first check is its baseline
        baseline = seen["was"] is None and name == "times"
        if happening and not seen["was"] and not baseline:
            seen["count"] += 1
        seen["was"] = happening
        return seen["count"] >= wanted

    if name == "and":
        return all(evaluate(t, s, history) for t in args)
    if name == "or":
        return any(evaluate(t, s, history) for t in args)
    if name == "not":
        return not evaluate(args[0], s, history)

    return bool(RELATIONS[name](s, *[evaluate_term(a, s, history) for a in args]))


def observe_history(trees, state: State, history: dict) -> None:
    """Sample temporal expressions in all branches, including inactive stages."""
    seen = set()

    def walk(node):
        if isinstance(node, list):
            for sub in node:
                walk(sub)
        elif isinstance(node, dict):
            name, args = unpack_expression(node)
            if name == "initially":
                evaluate(node, state, history)
                return
            for arg in args:
                walk(arg)
            if name in TEMPORAL:
                key = json.dumps(node, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    evaluate(node, state, history)

    # Observe every branch before boolean short-circuiting can hide events.
    for tree in trees:
        walk(tree)
