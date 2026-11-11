# Getting started with DexBench

[Back to DexBench](../README.md) · [Evaluation quickstart](../scripts/evaluation/eval_launch/QUICKSTART.md)

Run host commands from the repository root. Simulation commands run inside Docker.

## 1. Install

Prerequisites on the host: `git`, `python3`, the Hugging Face CLI
(`pip install -U huggingface_hub`), Docker with the NVIDIA container toolkit, and a
Hugging Face account that can read
[`RLWRLD/DexBench-assets`](https://huggingface.co/datasets/RLWRLD/DexBench-assets),
[`RLWRLD/DexBench-eval`](https://huggingface.co/datasets/RLWRLD/DexBench-eval)
and [`RLWRLD/DexBench-datasets`](https://huggingface.co/datasets/RLWRLD/DexBench-datasets).
Git LFS is not necessary.

**GPU.** The simulator renders every Run's cameras with Isaac Sim's RTX renderer,
so evaluation (and teleoperation) needs an NVIDIA GPU with RT cores: a GeForce
RTX, RTX PRO or L40S-class card with at least 16 GB (Isaac Sim 6.1's minimum is a
GeForce RTX 4080). Isaac Sim does not support GPUs without RT cores, such as
A100, H100 and H200. Those GPUs can still host the policy server, which the
simulator reaches over the network: `run_benchmark_cell.py --set remote_host=…`
for a ZeroMQ or websocket server, `--set remote_url=…` for an HTTP server.

Clone the repository. Then initialise only the two source checkouts that the
Docker image copies, IsaacLab-Arena and its Isaac Lab. The other nested checkouts
are not necessary:

```bash
# host
hf auth login

git clone git@github.com:RLWRLD/DexBench.git
cd DexBench
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init submodule/isaaclab_arena
GIT_LFS_SKIP_SMUDGE=1 git -C submodule/isaaclab_arena submodule update --init submodules/IsaacLab
```

## 2. Assets

The assets are not in Git. They are the Hugging Face dataset
[`RLWRLD/DexBench-assets`](https://huggingface.co/datasets/RLWRLD/DexBench-assets),
and `reproducibility/assets.lock.json` pins the commit of it that this checkout
uses. `scripts/download_assets.sh` downloads that commit into `assets/`, which Git
ignores:

```bash
# host, in the checkout
./scripts/download_assets.sh
```

The script then checks every file against the snapshot's receipt
(`assets/asset-upload-manifest.json`): the receipt must name the assets that the lock
pins, and each file must have the size and SHA-256 that the receipt records. A
second run downloads only the files that are missing or changed.

For part of the payload, name the folders, relative to `assets/`:
`./scripts/download_assets.sh --include "embodiments/simready/rby1m_wujihand2_beta2/**" --include "env/e4_hotel_fb/**"`.
The script then also downloads the small files that every part needs: the indexes,
reference views, sidecars and documents (about 7 MB; the receipt lists them with
`"storage": "git"`). An environment says which payload to download when an asset is
missing.

After a `git pull` that changes the lock, run the script again. If the new snapshot
no longer has a file, the check refuses it; `--prune` deletes such files. Run the
script on the host as your own user, never from a container: files a container
writes are root-owned. The dataset card, `assets/README.md`, lists the contents and
the license terms.

### Asset licensing

[`ASSET_LIST.md`](https://huggingface.co/datasets/RLWRLD/DexBench-assets/blob/main/ASSET_LIST.md)
(`assets/ASSET_LIST.md` after the download) records the identified contributor grants, exclusions
and generated texture sources. It is a partial attribution register, not a license
for the entire repository. Preserve package licenses and notices; unresolved
permissions remain unresolved.

## 3. Datasets

The data is in two Hugging Face datasets:

- [`RLWRLD/DexBench-eval`](https://huggingface.co/datasets/RLWRLD/DexBench-eval):
  the frozen start-state rows and layout sets that evaluation replays (`eval/db_conditions/`) and
  Hotel - Task 3's runtime inputs (`eval/db_h3_inputs/`).
- [`RLWRLD/DexBench-datasets`](https://huggingface.co/datasets/RLWRLD/DexBench-datasets):
  the training sets, one LeRobot v3.0 dataset for each of the 15 tasks (`db_r1/`
  to `db_h3/`) and one for all tasks (`multitask/`).

`reproducibility/datasets.lock.json` pins the revision of each, and
`scripts/download_datasets.sh` downloads at that revision. `--revision` overrides
the pin.

Evaluation needs the frozen start-state rows and their supporting inputs:

```bash
# host, in the checkout
./scripts/download_datasets.sh --eval-conditions
```

This command downloads `eval/db_conditions/` and `eval/db_h3_inputs/` from
`RLWRLD/DexBench-eval`. It checks
every row file against the frozen sets in
`source/dexbench_arena/dexbench_arena/benchmark/frozen/`, checks Retail - Task 1's layout sets
and Manufacturing - Task 5's piles against `benchmark/frozen/inputs.json`, and checks that Hotel - Task 3's
inputs give Hotel - Task 3's frozen rows. The check needs only `python3`. Then it installs
the files in the checkout, in `outputs/db_conditions/` and `outputs/db_h3_inputs/`.
It refuses to overwrite a local file that differs; `--force` replaces it.

Without `--eval-conditions`, the script downloads a repository into
`${DEXBENCH_DATASETS_ROOT:-$HOME/datasets}/dexbench_arena`: `RLWRLD/DexBench-eval`,
or the repository that `--repo-id` names. It gives its other arguments to
`hf download`, so `--include` selects a part. For example, the multitask training
set:

```bash
# host, in the checkout
./scripts/download_datasets.sh --repo-id RLWRLD/DexBench-datasets --include "multitask/**"
```

The `multitask` entry of the lock gives this repository's revision and this set's
path.

The episode caps are fixed data in the package:
`source/dexbench_arena/dexbench_arena/benchmark/demo_lengths/` holds the episode
lengths of each task's training demos, and `benchmark/spec.py` derives each cap
from them ([Eval_Protocol.md](../docs/Eval_Protocol.md), §8). Evaluation does not
need the training sets.

## 4. Docker

Isaac Sim is the runtime; every `/isaac-sim/python.sh` command below runs in the
container, from `/workspaces/dexbench_arena`. [docker/README.md](../docker/README.md)
is the Docker guide. Build the image from the two source checkouts of step 1:

```bash
# host, in the checkout
python3 docker/build.py
```

Then open a shell with [the run command](../docker/README.md#run). It mounts
`assets/` read-only and `outputs/`, where the frozen rows of step 3 are. The image
carries the committed source: commit a change and build again before you run it.
In the container, this checks the package:

```bash
/isaac-sim/python.sh -c "import dexbench_arena; dexbench_arena.register_components()"
```

The Kit-free tests run with `/isaac-sim/python.sh -m pytest source/dexbench_arena/tests`
in the container.

## 5. Evaluate a policy

The simulator needs an NVIDIA GPU with RT cores (**GPU** in §1). The policy server
can run on another machine.

The wrappers under `scripts/evaluation/` launch Arena's runners with DexBench's
environments added to Arena's environment registry. Call them, not the upstream
scripts: upstream does not know a `dexbench_`-prefixed environment.

One environment, one policy, one episode:

```bash
# container
/isaac-sim/python.sh scripts/evaluation/policy_runner.py \
    --viz none --policy_type zero_action --num_episodes 1 \
    dexbench_retail_v1_environment \
    --scene_layout fc_260821 --task retrieve_beverage_from_clutter --anchor Fridge \
    --embodiment dexbench_rby1_wuji2_joint
```

`--viz kit` opens a window to watch the run; it needs a display. The zero-action
policy holds still, so the episode ends at the task's time limit. Building the
scene takes most of the run. `--num_steps <n>` in place of `--num_episodes` stops
after `n` steps; a run that stops before an episode ends writes no episode result
and no video.

Outputs (videos with `--record_camera_video` / `--record_viewport_video`,
per-episode results, the report) land in a timestamped directory under
`--output_base_dir`. `--record_dataset` adds an Isaac Lab HDF5 dataset of every
episode's actions, observations and states; `--record_dataset_images` adds the
camera frames.

A typed Arena experiment, or a legacy `{"jobs": [...]}` JSON:

```bash
# container
/isaac-sim/python.sh scripts/evaluation/experiment_runner.py \
    --viz kit \
    --experiment_config scripts/evaluation/dexbench_zero_action_jobs_config.json
```

Check the benchmark pipeline before a model of your own: the eval launcher runs
the policy server kit's `HoldPose` (it holds the robot still) on Retail - Task 2's Train
cell, then scores it ([QUICKSTART.md](../scripts/evaluation/eval_launch/QUICKSTART.md),
path A). A model in the benchmark runs behind a policy server: a registered model
by its key, a model of your own by its client block (`--model-entry`,
[CONNECT_YOUR_POLICY.md](../scripts/evaluation/eval_launch/CONNECT_YOUR_POLICY.md)).

The benchmark: run one model on one cell from the cell's checked-in file, then
score the cells.

```bash
# container
/isaac-sim/python.sh scripts/evaluation/run_benchmark_cell.py --task <t> --cell <cell> --model <key> \
    --output outputs/db_<t>_eval_<cell>_<key> --checkpoint <label> -- --viz none
/isaac-sim/python.sh scripts/evaluation/score_benchmark_cells.py score --task <t> \
    --results train=<dir> ga_o=<dir> ... --json report.json
```

[Eval_Protocol.md](../docs/Eval_Protocol.md) describes the cells, how a task is
scored, and how to serve each policy.

When submissions open, submit leaderboard entries to the separate repository
[RLWRLD/DexBench-leaderboard](https://github.com/RLWRLD/DexBench-leaderboard). The
[submission guide](../docs/Leaderboard_Submission.md) defines the required results,
complete success/failure trajectories, videos and release-settings validation.

### Reference checkpoints

Reference checkpoints will be released on Hugging Face soon.

A report is `complete` only when every headline model ran every cell.
[Eval_Protocol.md](../docs/Eval_Protocol.md), §10, tells you how to read a report
without every headline model.

## 6. Teleoperate with a Meta Quest over OpenXR

Teleoperation needs two container shells: one runs the CloudXR server, the other
the simulation. Arena retargets the headset's poses and solves IK inside the scene.

Open the ports CloudXR needs on the host, once per machine (`48322/tcp` only in
HTTPS proxy mode):

```bash
# host
sudo ufw allow 49100/tcp   # signaling
sudo ufw allow 47998/udp   # media stream
sudo ufw allow 48322/tcp   # proxy, HTTPS mode only
```

Create one shared directory, then open both shells with
[the run command](../docker/README.md#run), each with
`--mount type=bind,src=/tmp/dexbench_cloudxr,dst=/cloudxr` added before the image
name ([CloudXR for teleoperation](../docker/README.md#cloudxr-for-teleoperation)):

```bash
# host
mkdir -p /tmp/dexbench_cloudxr
```

In the first shell, start the CloudXR server. It keeps the shell; start it before
the simulation and keep it running. The image contains the CloudXR profile that
the command reads:

```bash
# first shell - inside the container
/isaac-sim/python.sh -m isaacteleop.cloudxr \
  --cloudxr-install-dir /cloudxr --accept-eula \
  --cloudxr-env-config=/workspaces/dexbench_arena/docker/cloudxr_quest3.env
```

In the second shell, teleoperate. The example runs Retail - Task 2 (retrieve a drink from
the refrigerator clutter) in the retail_v1 store with the RBY1-M Wuji Hand 2:

```bash
# second shell - inside the container
cd /workspaces/dexbench_arena
source /cloudxr/run/cloudxr.env

/isaac-sim/python.sh scripts/imitation_learning/teleop.py \
    --viz kit --device cpu \
    dexbench_retail_v1_environment \
    --scene_layout fc_260821 --task retrieve_beverage_from_clutter --anchor Fridge \
    --teleop_device openxr \
    --embodiment dexbench_rby1_wuji2 --arm_mode dual_arm \
    --start_pose teleop_home --rby1_control_hz 30
```

Connect the headset, then press **Start AR** in the XR tab. `--arm_mode` takes
`left`, `right` or `dual_arm`. `--start_pose teleop_home` names the start pose
(the task's default otherwise), and `--rby1_control_hz 30` controls and records
the robot at 30 Hz, the rate a policy runs at.

To record demonstrations, use `record_demos.py` with the same arguments, an
output file and a target count. The CloudXR server must still run:

```bash
# second shell - inside the container
/isaac-sim/python.sh scripts/imitation_learning/record_demos.py \
    --viz kit --device cpu \
    --dataset_file outputs/my_demos.hdf5 \
    --num_demos 5 --num_success_steps 15 \
    dexbench_retail_v1_environment \
    --scene_layout fc_260821 --task retrieve_beverage_from_clutter --anchor Fridge \
    --teleop_device openxr \
    --embodiment dexbench_rby1_wuji2 --arm_mode dual_arm \
    --start_pose teleop_home --rby1_control_hz 30
```

`outputs/` is the directory the run command mounts from the host. The file keeps only successful episodes. Keep `--num_success_steps` near 15. With
1, a demonstration ends the first time the success condition is true, and for a
place task the object then still moves in the last frame. Such episodes rarely
replay.

## Local Python

For package-level checks only (imports, metadata, linting); Isaac Sim and
IsaacLab-Arena runtime work goes through Docker.

```bash
uv venv
uv pip install -e source/dexbench_arena
```
