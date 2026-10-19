"""Play tabletask episodes with pi-0.5 served by openpi over a websocket.

Start an openpi policy server with the `pi05_droid_jointpos` checkpoint first, then run:

    python -m examples.pi robolab-banana_in_the_named_bowl --episodes 20 --out build/pi
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

import tabletask
from tabletask.robots import Droid

HOST, PORT = "localhost", 8000  # openpi policy server address
CHUNK = 8  # actions executed per inference; openpi's open_loop_horizon for DROID
RATE = 1 / 15  # control period, seconds; DROID's 15 Hz control frequency used in training
SIZE = 224  # side of the square images sent to the policy, pixels


class Pi:
    """pi-0.5 client for openpi's websocket server, using DROID's observation and action format.

    Args:
        host: Policy server host.
        port: Policy server port.
        chunk: Actions executed from each inference before querying again.
    """

    def __init__(self, host=HOST, port=PORT, chunk=CHUNK):
        self.base = None
        from openpi_client import image_tools, websocket_client_policy

        self.fit = image_tools.resize_with_pad
        self.model = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.chunk, self.actions, self.index = chunk, None, 0

    def reset(self):
        """Discard the current action chunk."""
        self.actions, self.index, self.base = None, 0, None

    def act(self, observation, prompt):
        """Next joint-position action: 7 joint angles (radians) and gripper closure in [0, 1]."""
        if self.actions is None or self.index >= self.chunk:
            # Each action in a chunk is an offset from the joint positions sent with the query.
            self.base = np.asarray(observation["robot"]["joint_position"][:7], dtype=float)
            self.actions = self.model.infer(
                {
                    "observation/exterior_image_1_left": self.fit(
                        observation["images"]["external"], SIZE, SIZE
                    ),
                    "observation/wrist_image_left": self.fit(
                        observation["images"]["wrist"], SIZE, SIZE
                    ),
                    "observation/joint_position": np.asarray(
                        observation["robot"]["joint_position"][:7]
                    ).reshape(7),
                    "observation/gripper_position": np.asarray(
                        observation["robot"]["gripper"]
                    ).reshape(1),
                    "prompt": prompt,
                }
            )["actions"]
            self.index = 0
        action = np.asarray(self.actions[self.index])
        self.index += 1
        grip = float(np.clip(action[7], 0.0, 1.0))
        return np.concatenate([self.base + action[:7].reshape(7), [grip]])


def play(env, policy, *, seconds=45.0, out=None, every=4):
    """Play one episode and return its result.

    Args:
        env: An environment built with `Droid(action_space="joint_position")` and reset.
        policy: An object with `reset()` and `act(observation, prompt)`, such as `Pi`.
        seconds: Simulated time limit, in seconds.
        out: Directory for frames and `result.json`; None saves nothing.
        every: Save the external camera frame every this many steps.

    Returns:
        A dict with the task, episode, success, stop reason, steps and timings.
    """
    policy.reset()
    frames, step, began = [], 0, time.monotonic()
    reason = "time_limit"
    observation = env.observe()
    while env.status.time < seconds:
        if step % every == 0:
            frames.append(observation["images"]["external"].copy())
        action = policy.act(observation, env.task.instruction)
        observation, _, terminated, truncated, info = env.step(action, seconds=RATE)
        step += 1
        if terminated or truncated:
            reason = "solved" if info["success"] else (info.get("violated") or "ended")
            break
    frames.append(observation["images"]["external"].copy())  # final frame
    result = {
        "task": env.task.id,
        "episode": env.episode,
        "instruction": env.task.instruction,
        "success": bool(env.status.succeeded(env.task)),
        "stop_reason": reason,
        "steps": step,
        "simulated_seconds": round(env.status.time, 2),
        "wall_seconds": round(time.monotonic() - began, 1),
    }
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for index, rgb in enumerate(frames):
            Image.fromarray(rgb).save(out / f"{index:04d}.png")
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    """Parse the command line, play the requested episodes and write a summary."""
    parser = argparse.ArgumentParser(
        prog="python -m examples.pi", description="Drive a task with pi-0.5."
    )
    parser.add_argument("task", help="task id or task file")
    parser.add_argument("--episodes", type=int, default=1, help="number of episodes to play")
    parser.add_argument("--first", type=int, default=0, help="first episode number")
    parser.add_argument("--host", default=HOST, help="policy server host")
    parser.add_argument("--port", type=int, default=PORT, help="policy server port")
    parser.add_argument(
        "--seconds", type=float, default=45.0, help="simulated time limit per episode"
    )
    parser.add_argument("--chunk", type=int, default=CHUNK, help="actions executed per inference")
    parser.add_argument("--out", help="directory for frames and results")
    args = parser.parse_args(argv)

    out = Path(args.out or Path("build/pi") / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    policy = Pi(args.host, args.port, args.chunk)
    env = tabletask.make(args.task, robot=Droid(action_space="joint_position"))
    played = []
    try:
        for episode in range(args.first, args.first + args.episodes):
            env.reset(episode=episode)
            result = play(env, policy, seconds=args.seconds, out=out / f"episode_{episode:03d}")
            played.append(result)
            print(json.dumps(result), flush=True)
    finally:
        env.close()
    solved = sum(result["success"] for result in played)
    summary = {
        "task": args.task,
        "episodes": len(played),
        "solved": solved,
        "success_rate": round(solved / max(1, len(played)), 3),
        "out": str(out),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps({"summary": summary, "episodes": played}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
