# pt-ego-recovery

Bottleneck Recovery Replay (BRR) for π₀.₅ on the REBOOT recovery data. BRR scores each recorded recovery stage by how far the current policy is from the expert, picks whole recoveries whose weakest stage is worst, and trains them with a loss that weights stages instead of frames. The first task is USB-A insertion.

openpi is a submodule and stays unmodified. This repo uses its model, weight loader, optimizer, tokenizer and normalization. It replaces openpi's training script with its own loop, because the BRR loss needs per-sample losses grouped by recovery stage, and because a checkpoint here holds only the trainable parameters (1.9 GB: LoRA adapters, SigLIP and the action projections) instead of the full 11 GB train state.

## Setup

```bash
git clone --recurse-submodules https://github.com/triquang26/pt-ego-recovery.git
cd pt-ego-recovery
source env.sh
(cd third_party/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync)
uv pip install --python $PY --no-deps -e .
```

`env.sh` keeps every cache inside the repo. Hugging Face downloads are anonymous. The `pi05_base` checkpoint downloads on first use (12 GB).

## Pipeline

```bash
$PY -m ptego.prepare --task usb_a
$PY -m ptego.segments --root data/reboot/usb_a
$PY -m ptego.visualize segments
$PY -m ptego.experiment base
CUDA_VISIBLE_DEVICES=0 $PY -m ptego.experiment run B --seed 0
CUDA_VISIBLE_DEVICES=1 $PY -m ptego.experiment run E --seed 0
$PY -m ptego.visualize scores runs/usb_a/E_brr_selection/frac0.25/seed0/round0
$PY -m ptego.evaluate report runs/usb_a
```

| Module | Output |
| --- | --- |
| `prepare` | `data/reboot/<task>/`: low-dimensional arrays, one JPEG store per camera, `manifest.json` with duplicate groups and the train/val/test split |
| `segments` | recovery window and stages written into the manifest, DTW progress curves in `progress.json` |
| `policy` | REBOOT to openpi adapter, model creation, checkpoint save and load, action sampling |
| `scoring` | Energy Score per anchor, per stage and per recovery |
| `selection` | uniform, ES-chunk and BRR selection under a frame budget |
| `losses`, `mixture` | mean, stage-balanced and BRR losses; batch layout and the weighted objective |
| `hil` | the HIL-UMI offline baseline (advantage-conditioned ACBC) |
| `train` | training loop |
| `experiment` | base policy, runs A to G, three rounds of score, select, train |
| `evaluate` | offline metrics on the test split and the comparison table |
| `visualize` | segment review sheets and videos, stage score heatmaps |

## Data

`prepare` reads only the action, state and index columns of the REBOOT parquet files over HTTP. Those files are 37 GB per dataset, almost all depth. It then decodes the AV1 videos of three cameras: overhead (`cam_high`) and both wrists. Frames are resized with padding to 224×224 and stored as JPEG, about 3 GB for USB-A.

Each REBOOT episode is a fixed 30 s recording at 30 Hz. The trailing idle frames are cut, keeping 10 frames of margin. State and action are the 14 joint positions of the two WidowX AI arms, in REBOOT order. The policy keeps that meaning and order, and the actions stay absolute. Normalization uses quantiles computed on the train split only.

`prepare` checks for duplicates by comparing resampled action trajectories across all episodes. USB-A has none, so every episode is its own group. Groups are split 70/15/15 for each kind:

| | train | val | test |
| --- | --- | --- | --- |
| nominal | 42 | 9 | 9 |
| recovery | 42 | 9 | 9 |

## Recovery segments

REBOOT does not publish its phase or failure annotations, so `segments` estimates them, and the manifest marks each one `"source": "auto"`.

1. Each recovery episode is aligned with DTW to the five closest train nominal episodes. Alignment uses joint positions, with the grippers weighted up. The median matched position gives a nominal progress curve.
2. The repair is the interval where time passes but nominal progress does not: the maximum subarray of `0.5 - pace * Δprogress`. Its start is `repair_start`, its end is `rejoin`.
3. Inside the repair, `retract` ends when the arm is halfway to its farthest point from the failure state. `retry` starts at the last time the arm is halfway from the rejoin state. `realign` is everything between. If any stage would be shorter than a tenth of the window, the window is split into thirds.
4. `failure_phase` is `pick` when the repair starts before the midpoint between the nominal grasp and release, and `place` otherwise.

On USB-A the method finds a repair in all 60 recovery episodes, 200 frames on average. In the first 25 episodes the window covers the extra gripper open and close of a re-grasp. In the later ones it covers a stall during placement. `runs/usb_a/viz/segments/` has one review sheet per episode, with keyframes at each boundary from the overhead and right wrist cameras, the progress curve and the gripper trace, plus `overview.png` and a few labelled videos. To correct an episode, edit its `recovery` entry in the manifest and set `"source": "manual"`. Later runs of `segments` leave it alone.

Frames before `repair_start` include the operator deliberately causing the failure, so they are never trained as targets. The one exception is the negative advantage class of run D.

## BRR

Energy Score, as in HIL-UMI: for 10 action chunks `X` sampled from the policy and the expert chunk `y`, both normalized and 20 steps long, `ES = mean‖X − y‖ − ½ mean‖X − X'‖`. Anchors are taken every 6 frames inside each stage. `d[i,k]` is the mean over the anchors of stage `k` in recovery `i`.

Selection priority is `S[i] = ½ mean_k d[i,k] + ½ max_k d[i,k]`. Recoveries are taken in decreasing `S` until the frame budget is reached, separately for each failure phase, so pick and place keep their share. A selected recovery is used whole: retract, realign and retry.

The BRR loss for a recovery is `½ mean_k L[i,k] + ½ max_k L[i,k]`, where `L[i,k]` is the flow matching loss averaged over frames of stage `k`. Action chunks may cross stage boundaries.

A training batch has 18 samples:

| Term | Weight | Mean loss | Stage loss (BRR) |
| --- | --- | --- | --- |
| nominal | 0.5 | 6 frames | 6 frames |
| selected recovery | 0.4 | 6 frames, uniform over the selected recovery frames | 1 recovery × 3 stages × 2 frames |
| random recovery | 0.1 | 6 frames, uniform over all train recovery frames | 1 recovery × 3 stages × 2 frames |

The policy is rescored after every round, so the selection follows whatever the policy still gets wrong.

## Runs

All runs start from the same base policy (`pi05_base` with LoRA, 10,000 steps on nominal train). Each then runs 3 rounds of 3,000 steps with the same budget and step count.

| Run | Selection | Loss |
| --- | --- | --- |
| A nominal_sft | none, nominal only | flow matching |
| B uniform_recovery | random whole recoveries | flow matching |
| C es_chunk | highest-ES anchors, independently | flow matching |
| D hil_umi_offline | same as C | ACBC: nominal and selected chunks positive, pre-repair frames negative, prompt suffix `Advantage: positive.` or `Advantage: negative.` |
| E brr_selection | BRR | flow matching |
| F brr_loss | random whole recoveries | BRR |
| G brr_full | BRR | BRR |

D is HIL-UMI moved to a fixed offline pool, and its results should be labelled that way. The comparisons that matter are E against C (selecting whole recoveries by their weakest stage), G against E (the loss on top of selection) and G against D.

`evaluate` reports on test episodes, which are never scored for selection: Energy Score and normalized action MSE over all recovery anchors (every 5 frames), the same per stage, the worst stage of each recovery, and nominal anchors (every 15 frames). Intervals are 95% bootstrap over episodes. These numbers measure offline imitation of recovery. They are not rollout success.

## Compute

On an A100 80 GB, one training step at batch 18 takes about 1.7 s on one GPU and 1 s on two. Scoring takes about 0.3 s per anchor with 10 samples and 10 flow steps.
