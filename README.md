# pt-ego-recovery

π₀.₅ with LoRA, fine-tuned on ManiSkill demos through openpi, with the task prompt conditioned on an advantage label (`Advantage: positive|negative`). The first task is `StackCube-v1`.

openpi is a submodule and stays unmodified. `ptego.policy.configs` registers the ManiSkill configs into openpi's config table, and `ptego.policy.run` runs openpi's own scripts with them.

## Environments

There are two environments, because openpi (JAX) and ManiSkill (SAPIEN) pin different dependencies.

```bash
git clone --recurse-submodules https://github.com/triquang26/pt-ego-recovery.git
cd pt-ego-recovery
source env.sh
(cd third_party/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync)
uv pip install --python $OPENPI_PY --no-deps -e .
uv venv --python 3.11 .venv-sim
uv pip install --python $SIM_PY mani_skill torch matplotlib "imageio[ffmpeg]" -e . -e third_party/openpi/packages/openpi-client
```

`env.sh` keeps every cache inside the repo. It also points Vulkan and EGL at the NVIDIA userspace libraries in `$NVGL`, because the container has no system Vulkan driver.

## Pipeline

```bash
$SIM_PY -m ptego.sim.demos --count 100 --out data/raw/stackcube_initial
$SIM_PY -m ptego.visualize data/raw/stackcube_initial/plan.rgb.pd_ee_delta_pose.physx_cpu.h5 --out runs/viz/expert
$OPENPI_PY -m ptego.convert data/raw/stackcube_initial/plan.rgb.pd_ee_delta_pose.physx_cpu.h5 --repo-id local/stackcube_initial
cd third_party/openpi
$OPENPI_PY -m ptego.policy.run norm-stats --config-name pi05_maniskill_acbc_lora
$OPENPI_PY -m ptego.policy.run train pi05_maniskill_acbc_lora --exp-name stackcube_initial
$OPENPI_PY -m ptego.policy.run serve policy:checkpoint --policy.config pi05_maniskill_acbc_lora --policy.dir ../../checkpoints/pi05_maniskill_acbc_lora/stackcube_initial/9999
$SIM_PY -m ptego.sim.evaluate --out runs/eval/stackcube_initial
```

ManiSkill's motion planner produces the demos in `pd_joint_pos`. ManiSkill's `replay_trajectory` then converts them to `pd_ee_delta_pose` and records RGB from both cameras. A demo is kept only if the replay also succeeds.

## Interface

| Field | Content |
| --- | --- |
| `image` | base camera, 224×224 RGB |
| `wrist_image` | hand camera, 224×224 RGB |
| `state` | TCP position (3), rotation as the first two columns of the rotation matrix (6), gripper opening (1) |
| `actions` | `pd_ee_delta_pose` command: translation (3), rotation (3), gripper (1), each in [-1, 1] |
| `advantage` | 1 or 0, turned into the prompt suffix per sample, before tokenization |

The model gets the state as discrete tokens (`discrete_state_input=True`). Actions are already deltas, so the config applies no extra delta transform.

A full openpi checkpoint (base parameters and train state) is about 11 GB.
