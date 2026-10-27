## tabletask

A minimal, object-centric format to author tabletop manipulation tasks in simulation, from simple to complex.

![tasks](https://github.com/user-attachments/assets/4aa5a007-2f82-4546-a281-3325b335b4a6)

A task says where things start and what must become true:

```json
{
  "id": "robolab-banana_in_bowl",
  "instruction": "Pick up the banana and place it in the bowl.",
  "start": [
    ["on", "banana", "table"],
    ["on", "bowl", "table"]
  ],
  "goal": [
    ["inside", "banana", "bowl"]
  ]
}
```

![regions: a drawer, a mug handle, a conveyor belt, a crate, the compartments of a tea drawer, the slots of a dish rack, a pouring cup and the pegs of a mug tree](https://github.com/user-attachments/assets/8742a8dd-6889-49b4-808f-4ada43a58ecf)

Objects carry named regions that tasks place things in and check against.


### Installation

```sh
pip install tabletask
pip install "tabletask[control]"
pip install "tabletask[rtx]"
```


### Example

```python
import tabletask

env = tabletask.make("robolab-banana_in_bowl")
obs, info = env.reset(seed=0)

for _ in range(600):
    action = [*obs["robot"]["joint_position"], 0.0]
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print("success:", info["success"])
env.close()
```

Several environments at once, each in its own process and renderer:

```python
envs = tabletask.make("robolab-banana_in_bowl", num_envs=8, gpus=[0])
observations, infos = envs.reset(episodes=range(8))
observations, rewards, terminated, truncated, infos = envs.step(actions)
envs.close()
```

EGL environments reuse their OpenGL context across resets. Call `env.close()` or
`envs.close()` to release it.

### Per-episode tasks and replay

Override a task for one reset, capture its initial state, and replay it later:

```python
env = tabletask.make("pick-max")
obs, info = env.reset(seed=42, task="pick_cube_lift3cm", capture_reset_state=True)
start = info["reset_state"]
env.reset(state=start)   # Rebuild the captured scene and restore its initial physics.
env.reset(seed=42)       # Sample the configured pick-max task again.
env.close()
```

`reset(task=...)` accepts the same task forms as `make`: a task id, file, document,
or compiled `Task`. Optional `scene` and `bindings` overrides also last for one
reset. Omitted overrides use the configured defaults. Passing `bindings={}` clears
configured bindings for that episode.

`capture_reset_state=True` adds a detached initial `ResetState` to the reset info.
Ordinary resets and step infos omit this replay payload. Keep the captured state
if it is needed after stepping. It contains the concrete task, scene XML, initial physics,
and robot configuration and can be pickled. Asset paths must remain accessible,
and replay requires compatible simulator versions. With `state=...`, omit the seed
or use the captured seed. Scene and binding overrides are rejected because the
captured state fixes the physical scene.

For matched instructions, supply `reset(state=start, task=other_task)`, where
`other_task` is a concrete `Task` derived from `start.task` with a different
instruction and objective. It must retain the captured scene and asset root.
Each replay starts a fresh task monitor and records the restored poses as its
initial baseline. Change the instruction and success/failure conditions together.

Parallel workers accept one reset keyword dictionary per selected worker:

```python
envs = tabletask.make("pick-max", num_envs=2, gpus=[0])
observations, infos = envs.reset(reset_kwargs=[
    {"seed": 42, "capture_reset_state": True},
    {"seed": 43},
])
start = infos[0]["reset_state"]
envs.reset(reset_kwargs=[{"state": start}], envs=[1])
envs.reset(reset_kwargs=[{"seed": 5}, {"seed": 6, "task": "pick_cube_lift3cm"}])
envs.close()
```

`episodes` and `reset_kwargs` are mutually exclusive. Legacy integer episode lists
remain supported. Each dictionary is unpacked into the selected worker's `reset`.
Worker errors are reported to the caller after all dispatched replies are drained.

### Optional bounding-box labels

Request labels for named task objects in selected cameras. For `pick-max`, `item1`
is the target object:

```python
env = tabletask.make("pick-max", bounding_boxes={"external": ["item1"]})
obs, info = env.reset(seed=0)
target = obs["bounding_boxes"]["external"]["item1"]
box = target["yxyx"]       # int32 [ymin, xmin, ymax, xmax] in RGB image pixels.
label = target["label"]    # The target asset's description, as used in the instruction.
visible = target["visible"]
env.close()
```

- Labels are disabled by default. The same option works with `num_envs`, and
  `envs.set(bounding_boxes={"external": ["item1"]})` changes it for existing workers.
- Box labels reuse the episode's sampled object names. Instructions without object
  placeholders use the assets' main names.
- Boxes enclose visible object pixels, including child bodies, with exclusive maximum
  coordinates. Fully occluded or offscreen objects return an all-zero box and `visible=False`.
- Add `"wrist": ["item1"]` to label the wrist view. Each labeled camera adds one
  segmentation render. Labels are returned only when that camera's RGB image is rendered.
  Action chunks generate labels once at the final observation, alongside RGB.
- Coordinates match the returned image size. Resize, crop, or letterbox them with the
  image before converting to model-specific tokens. Raster rendering is required.
- For PaliGemma supervision, retain the yxyx order, quantize the transformed coordinates
  into its 1,024 location bins, and construct the answer as four location tokens followed
  by a space and the label. The prompt is `"detect " + label + "\n"`. Cache the prompt
  and label tokens per description. Keep invisible targets separate from valid boxes.


### Acknowledgments

Tasks are ported from [ABC](https://github.com/amazon-far/abc) and
[RoboLab](https://github.com/NVlabs/RoboLab). The robot models come from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

### Shared object pools

An object entry may use `"asset_pool": "../pools/example/manifest.json"` instead of `asset`.
The manifest contains `asset_count` and an ordered `selection` array of records with an `asset` name.
Paths resolve relative to the task file, or the working directory for dictionary and JSON-string inputs.
Empty pools, duplicate members, conflicting asset declarations, and mismatched counts are rejected.
Loading expands the pool to an ordinary asset list, so compiled tasks remain usable without the manifest.
