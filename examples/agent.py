"""Run a vision-language model on one tabletask episode through its provider's API.

The model sees the external and wrist cameras plus robot state, and moves the Droid arm with
Cartesian tool calls. Each run writes images, a call log and a result to a new directory.

    python -m examples.agent robolab-cube_in_the_bowl --api anthropic --model <name> --ask-key
    python -m examples.agent robolab-cube_in_the_bowl --dry-run
"""

import argparse
import getpass
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

import tabletask
from tabletask.robots import Droid

from .api import Model
from .tools import SYSTEM, TOOLS, RobotTools


def _write_json(path, value):
    """Write `value` to `path` as indented JSON."""
    path.write_text(json.dumps(value, indent=2) + "\n")


def _dry_action(turn, home_z):
    """Scripted action for --dry-run: lift 2 cm, return to the start height, then finish."""
    if turn > 2:
        return {"name": "done", "args": {"note": "Offline check finished."}}
    return {
        "name": "move_to",
        "args": {
            "targets": {"z": home_z + (0.02 if turn == 1 else 0)},
            "note": "Offline camera and motion check.",
        },
    }


def _check_positive(**values):
    """Raise ValueError unless every value is a positive int."""
    for name, value in values.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def _make_client(api, model, api_key, base_url, reasoning, max_tokens, dry_run):
    """Provider client for the run, or None for a dry run."""
    if not dry_run and not model:
        raise ValueError("Supply a model name")
    if dry_run:
        return None
    return Model(
        api,
        model,
        api_key=api_key,
        base_url=base_url,
        reasoning=reasoning,
        max_tokens=max_tokens,
    )


def _make_out_dir(out):
    """Create the new results directory, defaulting to `build/runs/<UTC timestamp>`."""
    out = Path(out or Path("build/runs") / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    return out


def _observe(robot, turn, max_calls, image_size, feedback, out):
    """Observation for one turn, with its camera images saved to `out`."""
    observation = robot.observe(max_calls - turn + 1, image_size)
    for name, rgb in observation["images"].items():
        Image.fromarray(rgb).save(out / f"{turn:03d}_{name}.png")
    if feedback is not None:
        observation["previous_execution"] = feedback
    return observation


def _play_turn(robot, client, observation, feedback, turn, home_z, dry_run, out):
    """Ask for one action, execute it, log the call and return the execution feedback."""
    started = time.monotonic()
    if dry_run:
        actions = [_dry_action(turn, home_z)]
    else:
        actions = client.ask(observation, feedback, SYSTEM, TOOLS)
    latency = time.monotonic() - started
    if len(actions) == 1:
        action = actions[0]
        feedback = robot.execute(action["name"], action["args"])
    else:
        feedback = {
            "error": "Issue exactly one tool call; no movement executed",
            "movement_executed": False,
        }
    record = {
        "turn": turn,
        "observation": {key: value for key, value in observation.items() if key != "images"},
        "actions": actions,
        "execution": feedback,
        "api_seconds": latency,
        "usage": client.usage if client else {},
    }
    with (out / "calls.jsonl").open("a") as log:
        log.write(json.dumps(record) + "\n")
    print(
        json.dumps({"turn": turn, "execution": feedback, "api_seconds": round(latency, 3)}),
        flush=True,
    )
    return feedback


def _stop_reason(feedback, robot):
    """Reason to stop after a turn, or None to continue."""
    if feedback.get("ended"):
        return feedback["ended"]
    if robot.ended:
        return "environment"
    return None


def run(
    env,
    *,
    api="openai",
    model=None,
    api_key=None,
    base_url=None,
    reasoning=None,
    max_calls=57,
    seconds=45.0,
    chunk_steps=None,
    image_size=224,
    max_tokens=4096,
    out=None,
    dry_run=False,
):
    """Run a model on one built environment and log every call.

    The caller owns the environment and closes it.

    Args:
        env: An environment built with `Droid(action_space="cartesian_pose")`.
        api: Provider: "openai", "anthropic" or "gemini".
        model: Provider model name, passed through unchanged. Required unless `dry_run`.
        api_key: Key to use instead of the provider's environment variable.
        base_url: Endpoint to use instead of the provider's default.
        reasoning: Provider reasoning effort or level; None leaves it unset.
        max_calls: Maximum number of model requests.
        seconds: Maximum simulated motion time, in seconds.
        chunk_steps: Control steps after which a move stops and returns to the model;
            None executes each move in full.
        image_size: Width and height of each camera image, in pixels.
        max_tokens: Maximum output tokens per response.
        out: New directory for results; defaults to `build/runs/<UTC timestamp>`.
        dry_run: Replace the model with a scripted lift-and-return, without API calls.

    Returns:
        The result dict also written to `result.json`.

    Raises:
        ValueError: If an argument is invalid or `model` is missing.
        FileExistsError: If `out` already exists.
    """
    _check_positive(max_calls=max_calls, image_size=image_size, max_tokens=max_tokens)
    robot = RobotTools(env, seconds=seconds, chunk_steps=chunk_steps)
    client = _make_client(api, model, api_key, base_url, reasoning, max_tokens, dry_run)
    out = _make_out_dir(out)
    config = {
        "task": env.task.id,
        "episode": env.episode,
        "api": api,
        "model": model,
        "reasoning": reasoning,
        "max_calls": max_calls,
        "seconds": seconds,
        "chunk_steps": chunk_steps,
        "control_dt": robot.dt,
        "image_size": image_size,
        "max_tokens": max_tokens,
        "dry_run": dry_run,
    }
    _write_json(out / "config.json", config)
    calls, feedback, reason = 0, None, "call_budget"
    began = time.monotonic()
    home_z = float(env.runtime.robot.observe()["tcp_pose"][2])
    try:
        for turn in range(1, max_calls + 1):
            if robot.ended:
                reason = "environment"
                break
            observation = _observe(robot, turn, max_calls, image_size, feedback, out)
            calls += 1
            feedback = _play_turn(robot, client, observation, feedback, turn, home_z, dry_run, out)
            stop = _stop_reason(feedback, robot)
            if stop is not None:
                reason = stop
                break
    except KeyboardInterrupt:
        reason = "interrupted"
        raise
    except Exception:
        reason = "error"
        raise
    finally:
        result = {
            "success": env.status.succeeded(env.task),
            "stop_reason": reason,
            "calls": calls,
            "simulated_seconds": env.status.time - robot.started,
            "wall_seconds": time.monotonic() - began,
            "out": str(out),
            "dry_run": dry_run,
        }
        _write_json(out / "result.json", result)
    return result


def main():
    """Parse the command line, build the environment and run one episode."""
    parser = argparse.ArgumentParser(
        prog="python -m examples.agent", description="Run a vision model on one tabletask episode."
    )
    parser.add_argument("task", help="task id or task file")
    parser.add_argument("--api", choices=["openai", "anthropic", "gemini"], default="openai")
    parser.add_argument("--model", help="provider model name")
    parser.add_argument(
        "--ask-key", action="store_true", help="read the API key from a hidden prompt"
    )
    parser.add_argument("--base-url", help="compatible endpoint instead of the provider default")
    parser.add_argument("--reasoning", help="provider reasoning effort or level")
    parser.add_argument("--episode", type=int, default=0, help="seeded start to play")
    parser.add_argument("--max-calls", type=int, default=57, help="maximum model requests")
    parser.add_argument("--seconds", type=float, default=45.0, help="maximum simulated motion time")
    parser.add_argument(
        "--chunk-steps", type=int, help="return to the model after at most this many control steps"
    )
    parser.add_argument("--image-size", type=int, default=224, help="camera image size, pixels")
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="maximum output tokens per response"
    )
    parser.add_argument("--out", help="new directory for results")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="test cameras and two small motions without an API call",
    )
    args = vars(parser.parse_args())
    if not args["dry_run"] and not args["model"]:
        parser.error("--model is required unless --dry-run is used")
    ask_key = args.pop("ask_key")
    if ask_key and not args["dry_run"]:
        api_key = getpass.getpass("API key (hidden): ").strip()
        if not api_key:
            parser.error("API key cannot be empty")
        args["api_key"] = api_key
    task, episode = args.pop("task"), args.pop("episode")
    env = tabletask.make(task, robot=Droid(action_space="cartesian_pose"))
    env.reset(episode)
    try:
        print(json.dumps(run(env, **args), indent=2))
    except KeyboardInterrupt:
        print("Stopped. The run result is marked interrupted.")
        raise SystemExit(130) from None
    finally:
        args.pop("api_key", None)
        env.close()


if __name__ == "__main__":
    main()
