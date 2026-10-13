# Examples

Three ways to drive a task. These scripts live next to `tabletask/` and are not part
of the installed package:

- `agent.py`: a vision-language model through its provider's API.
- `pi.py`: pi-0.5 through openpi.
- `teleop.py`: your own hand, through an iOS device.

Run them from the repository root (the directory containing `tabletask/`).

## Vision-language model

Load a task, attach Droid, and let a model control it through camera observations
and Cartesian motion tools.

### Install

```sh
python -m pip install -e ".[control]"   # from the repository root
```

`mink` and `daqp` provide Droid's Cartesian controller. The API clients use only
Python's standard library, so no provider SDK is needed. On a headless Linux machine:

```sh
export MUJOCO_GL=egl
```

### Run

Anthropic:

```sh
python -m examples.agent robolab-cube_in_the_bowl \
  --api anthropic --model claude-fable-5-1 --reasoning medium --ask-key
```

`--ask-key` reads the API key from a hidden prompt. The key stays in the process's
memory; it is not echoed, added to shell history, or written to the run files.

OpenAI and Gemini:

```sh
python -m examples.agent robolab-cube_in_the_bowl \
  --api openai --model gpt-6-astra --reasoning medium --ask-key

python -m examples.agent robolab-cube_in_the_bowl \
  --api gemini --model gemini-3.8-flash --reasoning medium --ask-key
```

Model names are passed through unchanged, and your account must have access to the
model. `--reasoning` is optional; omit it for models without that setting.

Without `--ask-key`, the key is read from the environment:

| `--api` | Environment variable | Endpoint |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | Messages |
| `openai` | `OPENAI_API_KEY` | Responses |
| `gemini` | `GEMINI_API_KEY` | generateContent |

**Stop with Ctrl+C.** The runner stops sending requests, closes the environment,
and records `stop_reason: "interrupted"`. A request already sent may still complete
and be billed. HTTP errors stop the run; requests are not retried.

### From Python

With the provider's key in the environment:

```python
import tabletask
from tabletask.robots import Droid
from examples.agent import run

env = tabletask.make("robolab-cube_in_the_bowl").add(Droid(action_space="cartesian_pose"))
env.reset(0)

try:
    result = run(
        env,
        api="anthropic",
        model="claude-fable-5-1",
        reasoning="medium",
        max_calls=30,
    )
    print(result)
finally:
    env.close()
```

`run()` also takes `api_key=` for a key obtained at runtime and `base_url=` for a
compatible endpoint. Each call to `run()` starts a new conversation. The caller owns
the environment and closes it.

### Options

Pass another task id or task file to change the task; `--episode 1` selects another
seeded start.

| CLI option | Python argument | Default | Meaning |
|---|---|---|---|
| `--max-calls` | `max_calls` | `57` | Maximum model requests |
| `--seconds` | `seconds` | `45` | Maximum simulated motion time, seconds |
| `--chunk-steps` | `chunk_steps` | whole move | Control steps before the model sees new images |
| `--image-size` | `image_size` | `224` | Width and height of each camera image, pixels |
| `--max-tokens` | `max_tokens` | `4096` | Maximum output tokens per response |
| `--out` | `out` | `build/runs/<timestamp>/` | New directory for results |

Without `--chunk-steps`, each move runs to completion before the next observation.
`--chunk-steps 8` (0.4 simulated seconds) or `--chunk-steps 16` (0.8 s) stops a move
early so the model can continue toward the same target or change course. Physics is
paused while the model responds.

The task's own time limit and `always` conditions also apply. The full conversation,
including images, reasoning items and thought signatures, is resent on every call.
Request size grows with each turn, so a call limit is not a cost limit.

### Observations and actions

Each request contains the task instruction, external and wrist RGB images, joint
positions, gripper closure, the world pose of the grasp point, and the remaining
budgets. **Object coordinates and goal conditions are not sent.** The model has no
code, filesystem or scene-query tools.

It has two motion tools:

```python
move_to(targets={"z": 1.05, "gripper": 0}, note="Move above the object, fingers open.")
move_by(deltas={"dz": -0.05}, note="Lower 5 cm towards the cube.")
```

| `move_to` field | Meaning |
|---|---|
| `x`, `y`, `z` | Absolute world position, metres; z up |
| `roll`, `pitch`, `yaw` | Radians, relative to the grasp orientation at the start |
| `gripper` | **0 open, 1 closed**, thresholded at 0.5 |

`move_by` takes offsets `dx`, `dy`, `dz` (metres) and `droll`, `dpitch`, `dyaw`
(radians) from the measured pose, plus the same absolute `gripper`. It lets a model
move in a direction it can see without knowing the world coordinate of the target.

Omitted position and angle fields keep their measured values; an omitted gripper keeps its last command. Orientation is
`Rz(yaw) Ry(pitch) Rx(roll) R_initial`. The gripper command applies during the
whole move, so close the gripper in a separate call after arriving.

Motion is interpolated at 0.15 m/s and 0.75 rad/s and tracked by Droid's Mink IK at
20 Hz. A gripper change takes at least 0.6 s, and every move ends with 0.2 s of
settling. Moves longer than 10 s are rejected. An IK error stops the move and reports
how many steps ran; paths are not checked for collisions in advance.
`chunk_complete` means every planned command was applied, not that the target was
reached; check the images and measured pose.

`done` and `give_up` end the episode. Success is evaluated by tabletask, not by the
model calling `done`.

### Results and dry runs

Each run writes:

| File | Contents |
|---|---|
| `config.json` | Task, episode, model and settings |
| `001_external.png`, `001_wrist.png`, ... | Images sent with each call |
| `calls.jsonl` | Observation, actions, execution feedback, latency and token usage per call |
| `result.json` | Success, stop reason, call count, simulated time and wall time |

`build/runs/` is gitignored. `--out` must not exist yet, so earlier runs are never
overwritten. API keys are never written to these files.

To check the cameras and controller **without calling a model**:

```sh
python -m examples.agent robolab-cube_in_the_bowl --dry-run
```

The dry run lifts the gripper 2 cm and lowers it again. It does not attempt the task.

See `python -m examples.agent --help` for all options. The code is split into
`api.py` (provider clients), `tools.py` (observations and motion tools) and
`agent.py` (episode loop and logging).

## pi-0.5

pi-0.5 is a vision-language-action policy: it takes the two camera images and the
arm's joint positions and returns joint-position actions at 15 Hz. Install the openpi
client with `pip install -e <openpi>/packages/openpi-client`, then serve the checkpoint
from an openpi checkout:

```sh
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid_jointpos \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_droid_jointpos
```

Then play the task:

```sh
python -m examples.pi robolab-banana_in_the_named_bowl --episodes 20 --out build/pi
```

Each episode writes frames and a `result.json` to its own directory under `--out`,
and `summary.json` holds the success rate. `--chunk` is the number of actions
executed from each inference before querying again; openpi's DROID client uses 8.

Two details the adapter must get right:

- **Actions are offsets from the joint positions sent with the query**, and every
  action in a chunk is relative to that same pose. They are neither absolute angles
  nor per-step deltas; treating them as absolute angles drives every joint toward
  zero. `pi05_droid` (without `jointpos`) differs again: it outputs joint velocities.
- **Images are resized with openpi's `resize_with_pad`** to 224 x 224 and sent under
  DROID's key names (exterior camera, then wrist).

## iOS device

Move an iOS device to move the gripper, one episode after another. Poses are streamed
over your network by [Teledex](https://github.com/omarrayyann/TeleDex) from the
device's ARKit tracking.

### Install

```sh
python -m pip install teledex mujoco        # add mink daqp for --robot droid
```

Install the app from the [App Store](https://apps.apple.com/ae/app/mujoco-ar/id6612039501).

### Run

```sh
python -m examples.teleop robolab-banana_in_bowl --episode 6
```

By default the robot is the floating `Gripper`, which has no arm, so every pose is reachable; pass
`--robot droid` to drive the full arm.

Scan the QR code printed in the terminal with the app to connect. A window shows the
`dashboard` camera, over the robot's shoulder, with the task instruction across the top. With
the full arm, only the gripper and the link that carries it are drawn. When the first
pose arrives, the phone's pose is mapped to the gripper's current pose, so nothing jumps.

Motion is in world axes regardless of how you hold the phone. Forward is the direction
the phone's camera faced when the app started tracking, so start the app facing the
same way as the robot (world +x).

| Phone | Robot |
|---|---|
| Move the device | The gripper follows, `--scale` metres per metre (default 2.0) |
| Rotate the device | The gripper rotates with it |
| Toggle | Fingers: **on closed, off open** |
| Button A (hold) | **Clutch**: the robot holds still while you reposition the phone |
| Button B | **Drop this episode** and start the next one, without counting it |

Use the clutch for long reaches: hold the button, bring the phone back to a comfortable
pose, release, and continue. The phone vibrates when the clutch engages and when an
episode ends.

When an episode ends (the goal is met, an `always` condition is violated, or time runs
out), the terminal prints the outcome, the next episode starts (`--episode` 6, then 7,
8, ...) and the phone is mapped to the gripper again. Success is evaluated by
tabletask. Close the window or press Ctrl+C to stop; nothing is recorded.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--robot` | `gripper` | `gripper` (the floating `Gripper`, no arm) or `droid` (the full arm) |
| `--episode` | `0` | First episode to play |
| `--scale` | `2.0` | Metres of gripper motion per metre of phone motion |
| `--port` | `8888` | Port the phone connects to |
| `--camera` | `dashboard` | Camera shown: a scene camera (`top`, `preview`) or a robot camera (`wrist`, `external`) |
| `--width` | `1280` | Window width in pixels |
| `--no-view` | off | Do not open the window |
| `--debug` | off | Print every goal check at each start and whenever one changes; draw the goal regions |

### From Python

```python
import tabletask
from tabletask.robots import Droid
from examples.teleop import teleoperate

env = tabletask.make("robolab-banana_in_bowl").add(Droid(action_space="cartesian_pose"))
env.reset(6)
try:
    solved, played = teleoperate(env, scale=2.0)
finally:
    env.close()
```

`teleoperate` accepts `phone=`: any object with `start`, `anchor_to`, `target`,
`clutching`, `restarting`, `buzz` and `stop`, so it can run without a device.
