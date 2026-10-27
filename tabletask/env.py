"""The environment: episodes of a task, with reset and step."""

from __future__ import annotations

import math
from copy import deepcopy
from numbers import Integral

from .language.format import load
from .language.monitor import Monitor, Status
from .language.schema import Task
from .privileged import privileged_state
from .robots import Robot
from .sampling import build, capture, labels, restore


class Env:
    """An episode of a task, with `reset` and `step` shaped like Gymnasium's.

    Make one with `tabletask.make(...)`, then `reset()` and `step()`.

    Attributes:
        task: The concrete task of the current episode (the template before `reset`).
        runtime: The simulator state of the current episode, or None before `reset`.
        status: Elapsed time, completed stages and any failure.
        cameras: Which robot cameras observations include: True for all, False for none,
            or a name or list of names.
        end_on_fall: Whether an object falling off the table ends the episode.
        image_size: Observation image size as (width, height).
        renderer: None to rasterise images, or "rtx" to path-trace them.
        reward: "success" or "subtasks", as `step` describes.
        bounding_boxes: Camera names mapped to sequences of task object names to label.
            Defaults to an empty dict. Only cameras included in the RGB observation
            are labeled, using raster rendering.
    """

    def __init__(
        self, task: Task, scene=None, bindings=None, *, monitor_interval=0.01, backend="mujoco"
    ):
        self.template = task
        self.task = task
        self.monitor_interval = monitor_interval
        self.backend = backend
        self.scene = scene
        self.bindings = bindings
        self._episode_template = task
        self.runtime = None
        self._render_context = None
        self.status = Status()
        self.episode = -1
        self.robot = None
        self.cameras = True
        self.end_on_fall = True
        self.image_size = (640, 368)  # 16:9, the shape of DROID's camera frames
        self.renderer = None
        self.reward = "success"
        self.bounding_boxes = {}
        self._object_labels = None

    @property
    def monitor_interval(self) -> float:
        """How often, in simulated seconds, task conditions are checked."""
        return self._monitor_interval

    @monitor_interval.setter
    def monitor_interval(self, seconds):
        if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("monitor_interval must be finite and positive")
        self._monitor_interval = seconds

    def add(self, robot: Robot) -> Env:
        """Add the robot that acts in the scene. Returns the environment, for chaining."""
        if self.runtime is not None:
            raise RuntimeError("add a robot before reset()")
        if self.robot is not None:
            raise ValueError("this environment already has a robot")
        if not isinstance(robot, Robot):
            raise TypeError("add() expects a Robot instance, such as Droid()")
        self.robot = robot
        return self

    def reset(
        self,
        episode=None,
        *,
        seed=None,
        task=None,
        state=None,
        scene=None,
        bindings=None,
        capture_reset_state=False,
        options=None,
    ):
        """Sample or replay an episode start. Omitted overrides use configured defaults.

        Args:
            episode: Sampling seed, default 0.
            seed: Alias for episode.
            task: Task id, path, document or Task for this reset. Replay requires the same scene.
            state: Captured ResetState to replay, using its seed.
            scene: Scene XML override for this reset.
            bindings: Asset bindings for this reset. An empty dict clears configured bindings.
            capture_reset_state: Include ResetState in info["reset_state"].
            options: Unused Gymnasium compatibility argument.

        Returns:
            (observation, info), with reset_state included in info only when requested.
        """
        if self.renderer not in (None, "rtx"):
            raise ValueError(f"unknown renderer {self.renderer!r}; use None or 'rtx'")
        if self.reward not in ("success", "subtasks"):
            raise ValueError(f"unknown reward {self.reward!r}; use 'success' or 'subtasks'")
        if state is not None and (scene is not None or bindings is not None):
            raise ValueError("state fixes the scene and bindings")
        episode = episode if seed is None else seed
        if state is not None:
            if episode is not None and episode != state.seed:
                raise ValueError("seed must match the captured reset state")
            if self.backend != state.backend:
                raise ValueError("reset state backend does not match the environment")
            episode = state.seed
        episode = 0 if episode is None else int(episode)
        try:
            self._close_runtime()
            if state is None:
                self._episode_template = self.template if task is None else load(task)
                if self.robot is not None and hasattr(self.robot, "start_pose"):
                    self.robot.start = self.robot.start_pose(episode)
                if self.robot is not None and hasattr(self.robot, "place"):
                    self.robot.place(episode)
                self.runtime = build(
                    self._episode_template,
                    episode,
                    self.scene if scene is None else scene,
                    self.bindings if bindings is None else bindings,
                    robot=self.robot,
                    backend=self.backend,
                )
            else:
                self._episode_template = deepcopy(state.template)
                self.runtime = restore(state, task)
            self._render_context = self.runtime.borrow_render_context(self._render_context)
            self.task = self.runtime.task
            self._object_labels = None
            self.episode = episode
            if self.renderer == "rtx":
                self.runtime.use_rtx(*self.image_size)
            self.monitor = Monitor(self.task)
            self.status = self.monitor.reset(self.runtime)
            self._best = self.status.progress
            info = self._info()
            if capture_reset_state:
                info["reset_state"] = capture(
                    self.runtime,
                    self._episode_template,
                    self.episode,
                    backend=self.backend,
                )
            return self.observe(), info
        except BaseException:
            self.close()
            raise

    def step(self, action=None, seconds=0.05):
        """Apply an action and simulate for `seconds`.

        Conditions are checked every `monitor_interval` seconds. Without a robot,
        call `step(seconds=t)` to simulate for `t` seconds.

        Args:
            action: The robot's action, in its action space; None to apply nothing.
            seconds: Simulated time to advance, rounded to whole physics steps.

        Returns:
            (observation, reward, terminated, truncated, info). The reward is 1.0 once
            the task has succeeded; with `reward` set to "subtasks", it is 1.0 at each step
            that first reaches a new milestone (see `Monitor`), and never paid twice. An
            episode terminates on success or on a broken always condition, and is truncated
            at the task's `time_limit`, if it has one.
        """
        self._require_reset()
        if self.robot is None and action is not None:
            raise ValueError("add a robot to step with an action; step(seconds=t) only simulates")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("seconds must be finite and nonnegative")
        if action is not None:
            self.runtime.robot.apply(action)
        timestep = self.runtime.model.opt.timestep
        remaining_steps = max(1, round(seconds / timestep)) if seconds > 0 else 0
        interval_steps = max(1, round(self.monitor_interval / timestep))
        while remaining_steps > 0 and not self.status.failed:
            steps = min(interval_steps, remaining_steps)
            before = float(self.runtime.data.time)
            self.runtime.advance(steps * timestep)
            elapsed = float(self.runtime.data.time) - before
            self.status = self.monitor.update(self.runtime, self.status.time + elapsed)
            if self.end_on_fall and not self.status.failed:
                fallen = self.runtime.fallen()
                if fallen:
                    self.status.violated = f"{fallen[0]} fell off the table"
            remaining_steps -= steps
        succeeded = self.status.succeeded(self.task)
        terminated = succeeded or bool(self.status.violated)
        truncated = bool(self.status.timed_out)
        reward = float(succeeded)
        if self.reward == "subtasks":
            reward = float(max(self.status.progress - self._best, 0))
            self._best = max(self._best, self.status.progress)
        return self.observe(), reward, terminated, truncated, self._info()

    def _info(self) -> dict:
        return {
            "time": self.status.time,
            "stages": list(self.status.done),
            "stages_total": len(self.task.goal.stages),
            "violated": self.status.violated,
            "success": self.status.succeeded(self.task),
            "episode": self.episode,
        }

    def render(self, **kwargs):
        """An image of the scene; keyword arguments go to `Runtime.picture`."""
        self._require_reset()
        return self.runtime.picture(**kwargs)

    def observe(self, cameras=None, *, width=None, height=None) -> dict:
        """The current observation.

        Args:
            cameras: Robot cameras to render: True for all, False for none, or names.
                Defaults to `env.cameras`.
            width: Image width in pixels. Defaults to `env.image_size[0]`.
            height: Image height in pixels. Defaults to `env.image_size[1]`.

        Returns:
            A dict with "time", "objects" (name -> (position, wxyz quaternion)), "robot"
            (the robot's own observation), "priv_combined" (the privileged state, see
            `tabletask.privileged`; single-arm robots only) and, when cameras are rendered,
            "images".
            With bounding_boxes enabled, also includes "bounding_boxes", mapping camera
            names to object names to records with "label", "visible", and "yxyx".
            yxyx is an int32 array of shape (4,) in image pixels, ordered as
            (ymin, xmin, ymax, xmax), with exclusive maximum coordinates. Invisible
            objects have visible=False and an all-zero box. Labels are omitted for
            cameras not rendered, including intermediate steps of action chunks.
        """
        self._require_reset()
        cameras = self.cameras if cameras is None else cameras
        width = self.image_size[0] if width is None else width
        height = self.image_size[1] if height is None else height
        runtime = self.runtime
        observation = {
            "time": self.status.time,
            "objects": {name: runtime.pose(name) for name in self.task.scene.objects},
        }
        if runtime.robot is not None:
            observation["robot"] = runtime.robot.observe()
            if "tcp_pose" in observation["robot"]:
                observation["priv_combined"] = privileged_state(
                    runtime, observation["robot"], self.status.time, self._episode_template
                )

        available = runtime.robot.cameras if runtime.robot is not None else ()
        if cameras is True:
            selected = available
        elif cameras is False:
            selected = ()
        elif isinstance(cameras, str):
            selected = (cameras,)
        else:
            selected = tuple(cameras)

        unknown = set(selected) - set(available)
        if unknown:
            raise ValueError(f"unknown robot cameras: {sorted(unknown)}")
        if selected:
            unknown_boxes = set(self.bounding_boxes) - set(available)
            if unknown_boxes:
                raise ValueError(f"unknown bounding-box cameras: {sorted(unknown_boxes)}")
            for dimension in (width, height):
                if (
                    not isinstance(dimension, Integral)
                    or isinstance(dimension, bool)
                    or dimension <= 0
                ):
                    raise ValueError("image width and height must be positive integers")
            observation["images"] = runtime.images(selected, width, height)
            requested = {
                camera: names
                for camera, names in self.bounding_boxes.items()
                if camera in selected and names
            }
            if requested:
                if self._object_labels is None:
                    self._object_labels = self.task.object_labels or labels(self.task)
                boxes = runtime.bounding_boxes(requested, width, height)
                for objects in boxes.values():
                    for name, box in objects.items():
                        box["label"] = self._object_labels[name]
                observation["bounding_boxes"] = boxes
        return observation

    def _close_runtime(self):
        """Close episode resources, retaining the render context."""
        runtime = self.runtime
        self.runtime = None
        if runtime is not None:
            runtime.close_renderer()

    def close(self):
        """Release episode resources and the retained render context."""
        try:
            self._close_runtime()
        finally:
            context = self._render_context
            self._render_context = None
            if context is not None:
                context.close()

    def _require_reset(self):
        if self.runtime is None:
            raise RuntimeError("call reset() before using the environment")
