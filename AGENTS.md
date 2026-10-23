# Working in this repository

TableTask is tabletop manipulation tasks written as JSON (`tasks/<tier>/`), the
objects they use as MuJoCo assets (`assets/objects/<name>.xml`, meshes in
`assets/mesh/<name>/`), and a library (`tabletask/`) that samples starts and checks
conditions. `examples/` runs tasks with a vision-language model, pi-0.5 or a phone.
Generated files go in the ignored `build/` directory.

Install the checkout with the control, mesh-import and lint extras (add `rtx` for
path tracing), and set `MUJOCO_GL=egl` when there is no display:

```sh
python -m pip install -e ".[all,dev]"
```

There is no test suite. `ruff format tabletask examples` and
`ruff check tabletask examples` must pass.

## The loop

Every change is: look at it, try it, check it. Nothing counts as working until you
have opened the picture and read the numbers.

```sh
python -m tabletask asset mug --mesh scan.glb --mass 0.3   # mesh -> assets/objects/mug.xml
python -m tabletask look assets/objects/bowl.xml           # six views, size, mass, regions
python -m tabletask look robolab-banana_in_bowl            # one start and every condition's value
python -m tabletask try robolab-banana_in_bowl --put banana=bowl/inside  # move, simulate, re-check
python -m tabletask try x --turn bottle=90,0,0 --joint laptop/lid=1.2 --put cup=table/left+0.05,0,0
python -m tabletask check tasks/base/x.json assets/objects/y.xml --build  # exit 1 on any problem
python -m tabletask preview x --samples 6                  # a grid of starts
python -m tabletask words                                  # the vocabulary
python -m tabletask schema                                 # the JSON Schema of a task file
```

`look` and `try` write `<name>.png` in the working directory (`--out` to choose).
`check --build` samples six seeded starts; `--json` prints machine-readable
results. `python -m tabletask check tasks assets/objects --build` checks the whole
library and takes minutes; run it before calling a change to tasks or assets done.

It establishes that every task samples six seeded starts, that the goal is false at
each of them, and that every asset passes its checks. **It does not establish that a
robot can do any of it**: nothing here measures finger access, reachability or
success rates.

## Code layout

- `language/format.py` reads task files and compiles them to a `Task`
  (`schema.py`); `grammar.py` checks conditions; `transformations.py` draws
  variables and counts and expands words, tags and quantifiers; `kernel.py` holds
  the built-in relations and terms, `evaluation.py` evaluates them, and
  `monitor.py` tracks stages over an episode. `vocabulary.py` has the defined
  words and quantifiers, `modules.py` loads `uses` modules,
  `errors.py` has the two exceptions, `rotation.py` has quaternion helpers.
- `sampling.py` draws an episode's concrete task and asks the simulator for a
  settled start; `env.py` runs episodes.
- `validation.py` checks tasks and assets; `inspection.py` explains a start
  condition by condition and draws previews.
- `mujoco/model.py` assembles a scene; `place.py` samples where objects start;
  `runtime.py` steps and queries the simulation; `geometry.py` has the hull,
  footprint and contact maths.
- `mujoco/importing.py` turns a mesh into an asset with colliders; `regions.py`
  edits placement regions; `surface.py` finds cavities and supports in a mesh;
  `checks.py` reports an asset's facts and faults.
- `mujoco/rtx.py` path-traces with OVRTX; `usd.py` exports the scene it renders;
  `appearance.py` and `sky.py` handle table and sky textures.
- `mujoco/robots/` has `Droid`, `Gripper` and the IK they use.

Public names are re-exported from `tabletask/__init__.py`; internal modules import
each other directly.

Everything that touches MuJoCo lives under `tabletask/mujoco/`, and nothing outside
it imports MuJoCo at module level, so `import tabletask` loads no simulator. A
second simulator would be a sibling folder exposing `build(task, rng, robot=None)`
and `CAPABILITIES`, listed in `sampling.BACKENDS`. Robots subclass
`tabletask.robots.Robot`.

Use module-level imports, except for the simulator, optional packages (RTX, mink,
mesh import) and real import cycles.

Style: a module starts with a one-line docstring saying what it is for; public
functions and classes document their arguments (Google style); every constant says
what it measures and in which units. Comments state facts and reasons, and say
"object", not "thing".

## Episodes

`tabletask.make(task)` takes a task id, a task file or a document; add a robot
with `.add(...)`, then `reset(episode)` and `step(action)`. The episode number is
the seed: `reset(3)` and `reset(seed=3)` give the same start, and the robot starts
at its home pose every time. `reset` returns `(observation, info)` and `step`
returns `(observation, reward, terminated, truncated, info)`. The reward is 1.0 once
every stage is done and 0.0 before; the stages done are in `info["stages"]`. An
episode terminates on success or when an always condition breaks, and is truncated
at the task's `time_limit` if it sets one (most do not), so cap the episode yourself.

`tabletask.make(task, num_envs=8, robot=Droid(), gpus=[0])` runs eight environments
in worker processes, one renderer each, as an `Envs` (`tabletask/vector.py`):
`reset(episodes)`, `step(actions)` and `observe()` take and return lists, one entry
per environment; an action may be a sequence of actions, run in turn with one render
at the end; `set(image_size=...)` and `get("task.instruction")` reach every worker.
Workers are spawned, so the caller's script needs a `__main__` guard.

The observation has `time`, `objects` (name -> position and wxyz quaternion),
`robot` and `images`. Set `env.cameras` (a name, a list, or `False`) and
`env.image_size` to change what `reset` and `step` render; `observe()` takes the
same arguments for one call. A step advances 0.05 s by default, and motion goes
through actuators, so reaching a target takes several steps.

`Env.monitor_interval` (default 0.01 s) is how often conditions are checked. It is
rounded to whole physics steps and does not change the physics timestep.

A stage does not complete while the robot touches an object the stage places, or
while that object is still moving: one that moved more than 2 mm, or whose
quaternion changed by more than 0.02, since the previous check. Only the objects a
condition places count (the first argument of `inside`, `on`, ...), and `touching`
places nothing, so a task can ask for contact made with an object still in hand.

An episode also ends when an object falls more than 10 cm below the table top;
`info["violated"]` names it. Set `env.end_on_fall = False` to keep going.

## Robots

`Droid(action_space=...)` takes `joint_position` (the default: seven joint angles
in radians), `joint_delta` (offsets) or `cartesian_pose` (a world TCP pose
`[x, y, z, qw, qx, qy, qz]`, solved with IK, which needs the `control` extra). All
append gripper closure: **0 open, 1 closed**. Observations hold `joint_position`,
`joint_velocity`, `gripper` and `tcp_pose`; cameras are `wrist` and `external`.
The IK holds only the arm, enforces joint limits and avoids no collisions;
unreachable targets raise `ValueError` before anything moves. The DROID model
adapts the Panda and Robotiq assembly from
[EgoGym](https://github.com/omarrayyann/EgoGym).

Each episode randomizes the Droid setup, seeded by the episode: joints start up to
`home_noise` (0.2 rad) from home; the base shifts up to `base_xy_noise` (5 cm) along x
and y and turns up to `base_yaw_noise` (0.15 rad); with chance `table_extension` (0.5)
the table top continues under the arm as a visual only, when the table is no higher than
the base. The external camera moves to the arm's other side with chance `camera_mirror`
(0.5), swings up to `camera_azimuth_noise` (45°) around the point it looks at, moves up
to `camera_distance_noise` and `camera_height_noise` (20 cm) and re-aims at that point
moved up to `camera_target_noise` (5 cm); its field of view changes up to
`camera_fov_noise` (4°). Set any of these to 0 to turn it off.

`Gripper()` is a floating parallel gripper with rigid tapered rib fingers and an
80 mm opening, for when reach and joint limits should not matter. It takes the same
eight values as `cartesian_pose`: the pose moves a mocap target the gripper is
welded to, so there is no IK and it arrives within a few steps. A large jump in
one step arrives just as fast and can throw what it holds, so move it in small
steps. `Gripper(count=2)` is a pair driven by 16 values, left then right.

A new robot subclasses `tabletask.robots.Robot` with `model`, `cameras`, `setup`,
`reset`, `apply` and `observe`. An environment has one robot, added before
`reset()`.

## Rendering

Images come from `runtime.images()` and `runtime.picture()`. By default they are
rasterised with `mujoco.Renderer`.

`make(task, renderer="rtx")` path-traces them with NVIDIA's OVRTX instead
(`pip install "tabletask[rtx]"`, an RTX GPU; `TABLETASK_RTX_GPU` picks GPUs).
`mujoco/rtx.py` exports the scene to USD once and renders the stage. If OVRTX or
the GPU is missing, the caller gets a `RuntimeWarning` naming what is missing and
the rasteriser is used; invalid render arguments still raise.

Rules for the RTX path:

- MuJoCo reports `cam_resolution` as 1x1, so the export writes a portrait
  aperture; `_aim` uses MuJoCo's view transform and frustum at the frame's aspect,
  including free and orthographic cameras.
- Creating the USD exporter writes meshes and textures (about 1 s); reposing takes
  8 ms. Keep one exporter per runtime and call `stage()` per frame.
- The exporter writes animation samples, not default poses: bake the current state
  into defaults and hide absent geoms explicitly. OVRTX 0.4+ uses OVStage ordinals
  and a write fence before rendering.
- Populate OVStage once, then update `omni:xform` and camera attributes through
  cached queries; reload only when appearance or scene structure changes, and
  release queries before repopulating. Live frames use four passes by default.
- The USD exporter connects only colour textures and treats shininess as
  metallic; `_material` transfers metallic and roughness values and the normal,
  roughness, metallic and occlusion maps. Data maps use raw colour space; normal
  maps need the USD scale and bias conversion.
- Lighting is a large window light and a weaker dome. `picture(skybox=...)` lights
  and surrounds the scene with a library skybox; `sky.py` converts the cube cross
  to a latlong texture. OVRTX already honours the stage's z-up axis.
- Keep cameras outside solid geometry; a path tracer sees darkness from inside an
  opaque object.
- To render many runtimes, reuse one `rtx.Session` serially, closing each runtime
  before the next, and close the session after the batch.
- The first frame compiles shaders and can take minutes. OVRTX 0.5 needs a C++
  runtime with `GLIBCXX_3.4.30`.

The OVRTX 0.5 and OVStage 0.2 setup follows
[Newton PR #4256](https://github.com/newton-physics/newton/pull/4256).

## The task file

Required: `id`, `instruction`, `start`, `goal`. Optional: `objects`, `fixtures`,
`appearance`, `assets`, `always`, `uses`, `states`, `variables`, `time_limit`.
Three more have defaults and are normally left out: `scene` (`main`), `library`
(`../assets`) and `tabletask` (the format version). Unknown keys are errors.

A condition is `[word, argument, ...]`, and a list of conditions means all of them.
`start`, the goal and `always` share one vocabulary. Lengths are metres, angles
degrees, quaternions wxyz, and the world is z-up. `left_of` is +y, `right_of` -y,
`in_front_of` -x (toward the robot), `behind` +x: world directions, not camera
ones.

`and`, `or` and `not` combine conditions; `at_least`, `at_most` and `between`
compare numbers, inclusively; `+`, `-` and `*` combine numbers; `tally` counts
true conditions. `every`, `some`, `no`, `every_pair` and `each` expand over a list
of names or a `#tag`, binding `it` (and `other` for pairs) unless a binder name
follows the body. `then` is a goal shorthand for stages in sequence.

### Objects

```json
"objects": {
  "piece": {"tags": ["pieces"], "count": [1, 3], "properties": {"graspable": true}},
  "container": {"properties": {"container": true}}
}
```

An object declares `asset`, `fixed`, `count`, `tags`, `properties`, `sample` and
`regions`. `asset` may be a list, a pool: each copy draws a different asset from it.
An instruction may name an object or copy, `{item1}`, to say one of its asset's labels (or
the asset's name with spaces): each episode picks one at random that no other object in the
scene shares, falling back to the main name, from a stream of its own so the words never
change where things are placed. `count` is an integer or an inclusive range; copies are
named `piece1`, `piece2`, ... (even when a range draws one) and keep the `piece` tag,
so conditions refer to them all as `#piece`; a start such as `on piece table` places
every copy (a counted object's start can only place it `on` or `inside` something and
turn it, without tolerances). `sample` is a region name, or a frame with `rel`, a
metric `offset`, `rotation` and `jitter`; it guides sampling, and the start conditions
still decide. A sample may name a region of the object an `on` or `inside` start names
bare (`table/left` for `on x table`); otherwise the sample must be the region the start
names. Without `sample` an object goes anywhere on `table/top`. `regions` adds regions an asset lacks. An object named
only in `start` uses the asset of its name; an object listed in `objects` must
name its `asset`, or it is abstract and needs a binding.

### Stages and time

A short goal is one stage that must hold for 0.5 s. Several goal entries make one
stage that must hold all at once; for steps in order use `then`:

```json
"goal": [["then", ["on", "brown_egg", "plate"], ["on", "white_egg", "plate"]]]
```

Explicit stages expose the timing:

```json
"goal": {"stages": {
  "fill": {"condition": ["inside", "egg", "bowl"], "maintain": true},
  "cover": {"condition": ["on", "lid", "bowl"], "after": ["fill"], "hold": 0.2}
}}
```

- `after` requires the earlier stages to have completed at an earlier check. It
  does not stop the condition becoming true early.
- `hold` counts consecutive true checks while the stage is eligible; 0 completes
  on the first.
- A completed stage stays completed. `maintain: true` requires the condition at
  every later check and fails the episode otherwise; `always` conditions apply
  from the first check.
- `once(c)` is whether `c` has ever been true, the start included; `times(c, n)` waits
  for `n` false-to-true transitions after the first check; `initially(c)` is whether `c`
  held at the start. They are tracked in every stage, active or not.

`uses` maps a namespace to a local module file and an exact version. Paths resolve
against the task file. Loading never downloads anything or runs code.

## Writing a task

A task is one JSON file in `tasks/<tier>/` (`base` or `safety`), and its
`id` is the file name. This is a whole task:

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

`start` says nothing about where things go; each episode draws a settled
arrangement that satisfies it. An object name resolves to
`assets/objects/<name>.xml`. Copy a similar task to begin.

After every edit, run:

```sh
python -m tabletask check tasks/base/mine.json --build                    # compiles? starts sample?
python -m tabletask preview tasks/base/mine.json --samples 6 --out p.png  # do the starts look right?
python -m tabletask look tasks/base/mine.json --seed 2 --out l.png        # one start, every condition
```

- `check --build` draws six seeded starts and rejects a goal already true at any
  of them. If sampling fails it counts why each draw was rejected, and
  `look --seed N` shows the last rejected draw with every condition's value.
- `preview` shows what validation cannot: an object too small to see, a layout
  that never varies, a container out of the robot's reach.
- `look` prints every condition with its value, so a false one shows the number
  against its threshold:

  ```
  goal  inside banana bowl, hold 0.5 s  FALSE
     contained(banana, bowl/inside, 0.01)  FALSE
  ```

Then put the objects in the goal state and confirm the goal becomes true:

```sh
python -m tabletask try tasks/base/mine.json --put banana=bowl/inside --settle 2
```

### The words

`python -m tabletask words` lists the vocabulary with every default.

| | |
|---|---|
| contact and containment | `on` `inside` `at` `near` `covering` `above` `below` `lifted` `touching` `within` `contained` |
| direction | `left_of` `right_of` `in_front_of` `behind` `facing` |
| arrangement | `aligned` `ordered` `count_in` |
| pose | `upright` `lying` `upside_down` `oriented` `oriented_like` |
| articulation | `joint` `open` `closed` |
| history | `at_start` |
| regions | `overlapping` |

### Pitfalls

- **A goal true at the start is rejected.** Give the episode something to do.
- **Positional words say nothing about support.** `left_of` compares positions, so
  an object held in mid-air satisfies it. Add the support:
  `[["left_of", "cube", "bowl"], ["on", "cube", "table"]]`.
- **`inside` tests the middle of the shape** and counts objects resting on
  something already inside; `on` is contact plus clearance.
- **Default tolerances are tight.** `at` allows 0.02 m from `b`'s centre; pass a larger
  one when the task allows it: `["at", "a", "b", 0.05]`. The object must still be within
  2 cm of `b`'s region, whatever the distance.
- **`touching` needs a real contact.** Two blocks 0.5 mm apart are not touching;
  use `overlapping` on regions to mean "these meet".
- **A region that dips below its object lifts it off the table.** `region` reports
  each floor height above the base; a negative one is the bug.
- **Every task must sample 6/6 starts.** Unreliable sampling means the start
  conditions and the geometry disagree.
- **The arm cannot reach the whole table.** Sample where it can, or the task is
  only solvable in some episodes.

Write task files with `tabletask.dumps` so every file has the same layout. The id
prefix records where a task came from: `robolab-` and `abc-` are ports,
and the rest were written here.

## Adding an asset

- `python -m tabletask asset` (needs the `importing` extra) imports a mesh: it
  orients it to rest on its base, generates convex collision hulls, distributes the
  mass across them, and proposes `inside`, `top` and `bounds` regions. Then `look`
  at it and fix the XML by hand where the proposal is wrong.
- `python -m tabletask asset --rehull <asset.xml> ...` regenerates an asset's
  collision hulls from its visual mesh, keeping the total mass and the collision
  attributes. `--jobs` runs several at once; CoACD is not deterministic, so a redo
  always changes the files.
- `python -m tabletask look <asset.xml> --collision` draws what physics collides
  with; mesh geoms are shown as the convex hulls they become.
- One file `assets/objects/<name>.xml`, one body named `<name>`, meshes in
  `assets/mesh/<name>/`: `0.obj` and `0.png` visual, `hulls/hNN.obj` collision
  and `inside.obj` from an import. Several assets can share a mesh folder, so
  `--rehull` writes `<asset>_hulls/` and `region --fit` writes
  `regions/<asset>_<region>.obj` to keep their files apart.
- Metres and z up in the XML, with the origin at the base: the object rests on the
  table at z=0. `look` reports the origin's height above the lowest point and the
  tilt after settling; both should be about 0.
- Every asset names itself in plain English, `<custom><text name="label" data="red mug"/></custom>`,
  distinct from every other asset's label; instructions and the dashboard use it. Other ways to
  say it follow, separated by `|`, from general to detailed: `data="red mug|mug|red cup|red
  ceramic mug|glossy red ceramic coffee mug"`. Each names this asset alone, never a different
  object; the first stays the main, distinct name.
- An asset's colour is fixed in its file, so its labels can say it. The same shape in several
  colours is several assets sharing one mesh folder, each painting its materials: `red_ceramic_bowl`,
  `blue_divided_tray`, `pink_ribbed_mug`, `green_dinner_plate`, `yellow_plastic_bucket` and
  `purple_plastic_jug` come in red, orange, yellow, green, blue, purple, pink and teal.
- An asset declares what tasks may use it for as `<text name="tags" data="pickable"/>` in the
  same `<custom>`, words separated by spaces: `pickable` (one hand grasps and lifts it; the
  pick-max pool) and `container` (a bowl, tray, plate or bin that things are put in). An asset
  with no tag is neither.
- Regions are geoms named `place:<region>` in class `place`, inside the body:
  `place:inside` (the cavity), `place:top` (where something rests on it) and
  `place:bounds` (a box around the visible mesh). Any other name is a custom
  region a task refers to as `<object>/<region>`.
- Materials are `<material>` with `<layer role="rgb|normal|occlusion|roughness|metallic">`.
- Table and sky textures are PNGs in `assets/looks/surface/` and `assets/looks/sky/`, listed
  by name in `tabletask/scenes/textures.json`; episodes draw only from that list, so add a
  new texture to both.
- Scene `appearance` also varies each episode: `table_height` (0.1 m) raises or lowers the
  table, and `lighting` (1; 0 turns it off) turns each light up to ~57° while keeping it
  aimed down, scales its brightness 0.4–1.6×, tints each colour channel up to ±0.15 and
  turns its shadows off with chance 0.5.

### Procedural assets

An object's asset may be a generator instead of a file:
`"asset": {"generate": "drawer", "params": {"width": [0.3, 0.4], "handle": ["knob"]}}`.
A generator is `assets/generators/<name>.py` with `PARAMS` (each parameter's fixed value,
`[low, high]` range or list of choices) and `build(params) -> str`, the MJCF of one asset
with one top-level body. A task's `params` narrow `PARAMS`. Each episode draws the values
from its own stream and `tabletask/generation.py` writes the asset to
`~/.cache/tabletask/generated/<name>-<hash>.xml` (`TABLETASK_GENERATED` moves it), naming
the model and body after the file, so sampling, conditions, regions and the privileged state
see an ordinary asset. A generator follows the asset conventions (front to -x, origin at the
base, a label, `place:` regions) and keeps its joint names and count fixed.
`assets/generators/drawer.py` builds a drawer in a box frame (joint `drawer`, regions
`inside`, `top`, `handle`) and `cabinet.py` a cabinet with one door hinged left or right
(joint `door`, regions `inside`, `shelf` when it has one, `top`, `handle`); both draw a
handle of five kinds placed anywhere on the front, and paint or library textures, from
`furniture.py`. `drawers-max` and `cabinet-max` use them.

Edit regions from the command line:

```sh
python -m tabletask region assets/objects/bowl.xml                       # list them
python -m tabletask region assets/objects/bowl.xml --out bowl.png        # and draw them
python -m tabletask region assets/objects/crate.xml \
  --add inside=box:0.18,0.12,0.09@0,0,0.06 --out crate.png               # add or replace
python -m tabletask region assets/objects/crate.xml --remove inside      # remove one
```

Extents are full width, depth and height in metres; `@x,y,z` places the centre in
the object's frame and `/roll,pitch,yaw` rotates it, in degrees. Each change
redraws the asset with its regions and reports each region's floor height above the
base. `--add` on an existing name replaces it, so the command can be repeated.

## Ported tasks and licenses

`abc-` tasks are ported from [ABC](https://github.com/amazon-far/abc) (source
commit `6c467ce`) and `robolab-` tasks from
[RoboLab](https://github.com/NVlabs/RoboLab). They are ports, not reproductions of
those benchmarks' scores: layouts, assets and episode distributions differ, and
goals use TableTask's 0.5 s hold. `assets/sources/NOTICE.md` and the license texts
beside it are required by the ported assets' licenses; keep them.
