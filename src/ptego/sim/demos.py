from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import mani_skill.envs
from mani_skill.examples.motionplanning.panda.run import MP_SOLUTIONS
from mani_skill.utils.wrappers.record import RecordEpisode

from ptego.schema import IMAGE_SIZE

ROBOT = "panda_wristcam"


def env_kwargs(control_mode: str) -> dict:
    return dict(
        robot_uids=ROBOT,
        obs_mode="none",
        control_mode=control_mode,
        render_mode="rgb_array",
        sensor_configs=dict(width=IMAGE_SIZE, height=IMAGE_SIZE),
        sim_backend="cpu",
    )


def plan(env_id: str, count: int, out: Path, seed: int) -> Path:
    env = gym.make(env_id, **env_kwargs("pd_joint_pos"))
    env = RecordEpisode(env, output_dir=str(out), trajectory_name="plan", save_video=False, record_reward=False)
    solve = MP_SOLUTIONS[env_id]
    successes, attempt = 0, seed
    while successes < count:
        result = solve(env, seed=attempt)
        ok = result != -1 and bool(result[-1]["success"].item())
        env.flush_trajectory(save=ok)
        successes += ok
        attempt += 1
        print(f"seed {attempt - 1} success {ok} total {successes}", flush=True)
    env.close()
    return out / "plan.h5"


def replay(trajectory: Path, procs: int) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mani_skill.trajectory.replay_trajectory",
            "--traj-path",
            str(trajectory),
            "--target-control-mode",
            "pd_ee_delta_pose",
            "--obs-mode",
            "rgb",
            "--save-traj",
            "--num-envs",
            str(procs),
            "--use-first-env-state",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="StackCube-v1")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--procs", type=int, default=8)
    args = parser.parse_args()
    replay(plan(args.env, args.count, args.out, args.seed), args.procs)


if __name__ == "__main__":
    main()
