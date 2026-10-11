from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import imageio.v3 as iio
import mani_skill.envs
import numpy as np
from openpi_client import websocket_client_policy
from PIL import Image, ImageDraw

from ptego.schema import IMAGE_SIZE, TASKS, state_vector
from ptego.sim.demos import ROBOT


def to_numpy(value) -> np.ndarray:
    return value[0].cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)[0]


def policy_input(obs: dict, task: str, positive: bool) -> dict:
    qpos = to_numpy(obs["agent"]["qpos"])
    return {
        "observation/image": to_numpy(obs["sensor_data"]["base_camera"]["rgb"]),
        "observation/wrist_image": to_numpy(obs["sensor_data"]["hand_camera"]["rgb"]),
        "observation/state": state_vector(to_numpy(obs["extra"]["tcp_pose"]), qpos[-2:].sum()),
        "prompt": task,
        "advantage": np.array([float(positive)], dtype=np.float32),
    }


def annotate(frame: dict, step: int, chunk: int, success: bool) -> np.ndarray:
    image = Image.fromarray(np.concatenate([frame["observation/image"], frame["observation/wrist_image"]], 1))
    ImageDraw.Draw(image).text((4, 4), f"t={step} chunk={chunk} success={success}", fill=(255, 255, 0))
    return np.asarray(image)


def rollout(env, client, task: str, seed: int, execute: int, positive: bool, max_steps: int) -> dict:
    obs, _ = env.reset(seed=seed)
    frames, chunk, success, step = [], 0, False, 0
    while step < max_steps and not success:
        inputs = policy_input(obs, task, positive)
        actions = client.infer(inputs)["actions"]
        for action in actions[:execute]:
            frames.append(annotate(inputs, step, chunk, success))
            obs, _, terminated, truncated, info = env.step(action[None])
            success = bool(to_numpy(info["success"]))
            step += 1
            inputs = policy_input(obs, task, positive)
            if success or step >= max_steps:
                break
        chunk += 1
    frames.append(annotate(inputs, step, chunk, success))
    return {"seed": seed, "success": success, "steps": step, "frames": frames}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="StackCube-v1")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--execute", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--negative", action="store_true")
    parser.add_argument("--videos", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    env = gym.make(
        args.env,
        robot_uids=ROBOT,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        sensor_configs=dict(width=IMAGE_SIZE, height=IMAGE_SIZE),
        max_episode_steps=args.max_steps,
    )
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(args.episodes):
        result = rollout(env, client, TASKS[args.env], args.seed + i, args.execute, not args.negative, args.max_steps)
        if i < args.videos:
            iio.imwrite(args.out / f"seed_{result['seed']}.mp4", np.stack(result.pop("frames")), fps=20, codec="libx264")
        else:
            result.pop("frames")
        results.append(result)
        print(json.dumps(result), flush=True)
    summary = {"success_rate": float(np.mean([r["success"] for r in results])), "episodes": results}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("success_rate", summary["success_rate"])


if __name__ == "__main__":
    main()
