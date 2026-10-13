"""`tabletask.make(..., num_envs=n)`: parallel environments in worker processes."""

import numpy as np

import tabletask
from tabletask.robots import Gripper

TASK = "robolab-banana_in_bowl"


def test_envs_reset_step_and_chunks():
    envs = tabletask.make(TASK, num_envs=2, robot=Gripper())
    try:
        envs.set(image_size=(64, 48))
        observations, infos = envs.reset(episodes=[0, 1])
        assert len(observations) == 2 and [i["episode"] for i in infos] == [0, 1]
        assert observations[0]["images"]["wrist"].shape == (48, 64, 3)
        assert envs.get("task.instruction") == [tabletask.load(TASK).instruction] * 2

        hold = [[*o["robot"]["tcp_pose"], 0.0] for o in observations]
        observations, rewards, terminated, truncated, infos = envs.step(hold, seconds=0.1)
        assert rewards == [0.0, 0.0] and not any(terminated) and not any(truncated)

        # a chunk of four actions for the first env only; the second is left alone
        chunk = np.repeat([hold[0]], 4, axis=0)
        observations, rewards, terminated, truncated, infos = envs.step([chunk, None], seconds=0.1)
        assert observations[1] is None and rewards[1] is None
        assert rewards[0].shape == (4,) and terminated[0].shape == (4,)
        assert abs(infos[0]["time"] - 0.5) < 1e-6

        # a partial reset
        observations, infos = envs.reset(episodes=[5], envs=[1])
        assert len(observations) == 1 and infos[0]["episode"] == 5
    finally:
        envs.close()


if __name__ == "__main__":
    test_envs_reset_step_and_chunks()
    print("ok")
