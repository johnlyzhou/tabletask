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
from tabletask.robots import Droid

env = tabletask.make("robolab-banana_in_bowl").add(Droid())
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
envs = tabletask.make("robolab-banana_in_bowl", num_envs=8, robot=Droid(), gpus=[0])
observations, infos = envs.reset(episodes=range(8))
observations, rewards, terminated, truncated, infos = envs.step(actions)
envs.close()
```


### Acknowledgments

Tasks are ported from [ABC](https://github.com/amazon-far/abc) and
[RoboLab](https://github.com/NVlabs/RoboLab). The robot models come from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
