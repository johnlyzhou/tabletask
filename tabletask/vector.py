"""Several environments for one task, each in its own process with its own renderer."""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
from contextlib import suppress

import numpy as np

_EGL_CUDA_DEVICE_NV = 0x323A


def egl_devices(gpus) -> list[int]:
    """The EGL device index of each CUDA GPU index, as MUJOCO_EGL_DEVICE_ID counts them.

    EGL numbers the GPUs in its own order, so a CUDA index cannot be used directly. The
    query runs in a subprocess so this process never loads EGL itself.
    """
    probe = f"""
from ctypes import *
import ctypes.util, json
egl = CDLL(ctypes.util.find_library("EGL"))
egl.eglGetProcAddress.restype = c_void_p
egl.eglGetProcAddress.argtypes = [c_char_p]
query_devices = CFUNCTYPE(c_bool, c_int, POINTER(c_void_p), POINTER(c_int))(
    egl.eglGetProcAddress(b"eglQueryDevicesEXT"))
query_attrib = CFUNCTYPE(c_bool, c_void_p, c_int, POINTER(c_long))(
    egl.eglGetProcAddress(b"eglQueryDeviceAttribEXT"))
devices, count = (c_void_p * 16)(), c_int()
query_devices(16, devices, byref(count))
cuda = {{}}
for i in range(count.value):
    value = c_long(-1)
    query_attrib(devices[i], {_EGL_CUDA_DEVICE_NV}, byref(value))
    cuda[i] = value.value
print(json.dumps(cuda))
"""
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    out = subprocess.check_output([sys.executable, "-c", probe], env=env, text=True)
    by_cuda = {cuda: int(egl) for egl, cuda in json.loads(out).items()}
    missing = [g for g in gpus if g not in by_cuda]
    if missing:
        raise ValueError(f"no EGL device for GPU {missing}; EGL has {sorted(by_cuda)}")
    return [by_cuda[g] for g in gpus]


def _trim_memory() -> None:
    """Hand freed simulator memory back to the OS (a worker otherwise grows every episode)."""
    gc.collect()
    libc = ctypes.util.find_library("c")
    if libc and sys.platform.startswith("linux"):
        ctypes.CDLL(libc).malloc_trim(0)


def _worker(pipe, source, robot, options) -> None:
    from . import make

    env = make(source, robot=robot, **options)
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                result = env.reset(data)
                _trim_memory()
            elif command == "step":
                actions, seconds = data
                result = _step(env, actions, seconds)
            elif command == "observe":
                result = env.observe(**data)
            elif command == "get":
                result = env
                for part in data.split("."):
                    result = getattr(result, part)
            elif command == "set":
                for name, value in data.items():
                    setattr(env, name, value)
                result = None
            elif command == "close":
                env.close()
                pipe.send(None)
                break
            pipe.send(result)
    except (EOFError, BrokenPipeError):
        env.close()


def _step(env, actions, seconds):
    """One action, or a sequence of them with cameras rendered only after the last."""
    actions = np.asarray(actions, dtype=float)
    if actions.ndim < 2:
        return env.step(actions, seconds)
    cameras = env.cameras
    env.cameras = False
    rewards = np.zeros(len(actions))
    terminated = np.zeros(len(actions), dtype=bool)
    truncated = np.zeros(len(actions), dtype=bool)
    try:
        for i, action in enumerate(actions):
            _, rewards[i], terminated[i], truncated[i], info = env.step(action, seconds)
            if terminated[i] or truncated[i]:
                break
    finally:
        env.cameras = cameras
    return env.observe(), rewards, terminated, truncated, info


class Envs:
    """`num_envs` environments for one task, each in its own process with its own renderer.

    Episodes are stepped together: `step` sends every environment its action and returns
    when all have moved. An action may also be a sequence of actions (shape `(k, dim)`),
    which the worker applies in turn and renders only once, at the end: the right shape for
    a policy that emits action chunks.

    Workers are started with the "spawn" method, so the calling script must be importable
    (its top level guarded by `if __name__ == "__main__":`), and anything that must not run
    again in a worker, such as creating a GPU backend for a learning framework, belongs
    under that guard or must be disabled for child processes.

    Args:
        source: The task, as `make` accepts it (a task id, file, JSON string or dict).
        num_envs: How many environments to run.
        robot: The robot each environment gets, as `make` accepts it.
        gpus: CUDA GPU indices to render on, one per worker in turn; None renders on
            whatever MUJOCO_EGL_DEVICE_ID or the driver chooses.
        **options: Passed to `make` (`renderer`, `scene`, `bindings`, ...).
    """

    def __init__(self, source, num_envs: int, *, robot="droid", gpus=None, **options):
        if not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs = num_envs
        devices = egl_devices(list(gpus)) if gpus else []
        context = mp.get_context("spawn")
        self._pipes, self._processes = [], []
        saved = {k: os.environ.get(k) for k in ("MUJOCO_GL", "MUJOCO_EGL_DEVICE_ID")}
        os.environ.setdefault("MUJOCO_GL", "egl")
        try:
            for i in range(num_envs):
                if devices:
                    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(devices[i % len(devices)])
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker, args=(child, source, robot, options), daemon=True
                )
                process.start()
                child.close()
                self._pipes.append(parent)
                self._processes.append(process)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def __len__(self) -> int:
        return self.num_envs

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _ask(self, command, data):
        for pipe in self._pipes:
            pipe.send((command, data))
        return [pipe.recv() for pipe in self._pipes]

    def set(self, **attributes) -> None:
        """Set attributes on every environment, such as `image_size` or `cameras`."""
        self._ask("set", attributes)

    def get(self, name: str) -> list:
        """An attribute of each environment, by dotted name: `get("task.instruction")`."""
        return self._ask("get", name)

    def reset(self, episodes=None, envs=None):
        """Start an episode in each environment (or only those in `envs`).

        Args:
            episodes: One episode number per environment reset, or None for random ones.
            envs: Indices of the environments to reset; None for all.

        Returns:
            (observations, infos): lists, one entry per environment reset.
        """
        envs = list(range(self.num_envs)) if envs is None else list(envs)
        if episodes is None:
            episodes = [random.randrange(2**31) for _ in envs]
        episodes = [int(e) for e in episodes]
        if len(episodes) != len(envs):
            raise ValueError(f"{len(envs)} environments reset but {len(episodes)} episodes given")
        for i, episode in zip(envs, episodes):
            self._pipes[i].send(("reset", episode))
        results = [self._pipes[i].recv() for i in envs]
        return [r[0] for r in results], [r[1] for r in results]

    def step(self, actions, seconds=0.05):
        """Apply each environment's action (or action sequence) and simulate `seconds` per action.

        Args:
            actions: One per environment: an action, a sequence of actions, or None to leave
                that environment alone this call.
            seconds: Simulated time per action, as in `Env.step`.

        Returns:
            (observations, rewards, terminated, truncated, infos): lists with one entry per
            environment, None for those given no action. For an action sequence, the
            rewards, terminated and truncated entries are arrays over the sequence, zero
            after the episode ended, and the observation is from after the last action run.
        """
        active = [i for i, action in enumerate(actions) if action is not None]
        for i in active:
            self._pipes[i].send(("step", (actions[i], seconds)))
        results = [None] * self.num_envs
        for i in active:
            results[i] = self._pipes[i].recv()
        return tuple([None if r is None else r[k] for r in results] for k in range(5))

    def observe(self, **kwargs) -> list:
        """The current observation of every environment; keyword arguments as in `Env.observe`."""
        return self._ask("observe", kwargs)

    def close(self) -> None:
        """Close environments concurrently and wait for every worker to exit."""
        pending = []
        for pipe in self._pipes:
            try:
                pipe.send(("close", None))
            except (EOFError, BrokenPipeError, OSError):
                pass
            else:
                pending.append(pipe)

        # Let renderer cleanup overlap before collecting acknowledgements.
        for pipe in pending:
            with suppress(EOFError, BrokenPipeError, OSError):
                pipe.recv()
        for pipe in self._pipes:
            pipe.close()

        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        self._pipes, self._processes = [], []
