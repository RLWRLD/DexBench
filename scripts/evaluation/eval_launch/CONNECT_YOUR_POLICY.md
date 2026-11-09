# Connect your policy to the evaluation

This guide takes you from "I have a trained model" to "I have a DexBench score". You run your model as a
small server. The eval launcher starts it next to the simulators, asks it for actions over HTTP, and scores
the episodes. You don't change the simulator or the benchmark. In the easy path, your whole change is one
config: it starts your server and registers your model.

```
 eval launcher worker (one per GPU)                          your server (your environment, any framework)
 ┌──────────────────────────────────────┐   GET  /info      ┌──────────────────────────────────────────┐
 │ starts your server from the recipe   │ ────────────────▶ │ what the model is: joints, horizon,      │
 │ runs the frozen start states         │                   │ image size, cameras                      │
 │ renders 4 cameras at 640x360         │   POST /predict   │                                          │
 │ asks for actions at 30 Hz,           │ ────────────────▶ │ images + joint state + instruction       │
 │ runs 16 of each chunk                │ ◀──────────────── │   → a chunk of absolute joint targets    │
 └──────────────────────────────────────┘                   └──────────────────────────────────────────┘
```

> **Hardware: use NVIDIA L40S GPUs (46 GB).** This pipeline was verified on L40S only: every launch, rehearsal and
> measured cost so far ran on L40S nodes (AWS g6e, one GPU per worker), the local backend included (a 4-GPU pod). The unit costs in `timings/` and the per-GPU memory budget (one simulator and its policy server
> sharing a GPU) are L40S figures. Other GPUs have not been tested: a card with less memory can starve the renderer
> (black frames), and a different card changes the costs the planner uses.

Five steps: write the server (1), register the model in your config (2), check two episodes (3), launch every
cell (4), score (5). Step 1 is the only code.

## What your model sees and answers

| | |
|---|---|
| **Robot** | RBY1-M with two Wuji2 hands: 62 action joints (torso 6, two 7-DoF arms, two 20-DoF hands, neck 2). |
| **Actions** | **absolute** joint positions, in radians, in the order *you* state. The client maps them to the robot by name. |
| **State** | the 62 measured joint positions, in your order. |
| **Images** | up to 4 cameras, uint8 RGB: `camera_ego_left`, `camera_ego_right` (head), `camera_wrist_left`, `camera_wrist_right` (palms). They are rendered at 640x360, and you choose your size (`frame_hw`, the same 16:9 aspect, no upscaling). The client area-downscales them before sending them as PNG. |
| **Instruction** | the task's text, e.g. "Take the can in the fourth lane from the left out of the open refrigerator's second shelf …". |
| **Rate** | one action per 30 Hz policy step. Each answer is a chunk, and the benchmark executes its first **16** actions before it asks again, so answer at least 16. |
| **Episodes** | every request stands alone: the client sends no episode id, and keeps nothing between requests. |

Train on data recorded at 30 Hz with the same cameras, and resize your training images with the same area
downscale.

## 1. Write the server

Use the server kit, `source/dexbench_policy_server` (the `dexbench-policy-server` package, version 0.1.0). It needs
Python 3.10+, numpy and Pillow, nothing from Isaac Sim or this repo, so it installs into your model's own
environment: `pip install <checkout>/source/dexbench_policy_server`. Subclass `ServedPolicy`:

```python
import json
from importlib.resources import files

import numpy as np
from dexbench_policy_server import ServedPolicy

# The robot's 62 action joints in the benchmark's order, from the kit (the table below).
ROBOT_JOINTS = json.loads(files("dexbench_policy_server").joinpath("rby1_wuji2_joint_names.json").read_text())


class MyPolicy(ServedPolicy):
    def load(self, checkpoint, options):                  # once, before the ready line
        self.policy = load_my_checkpoint(checkpoint)      # yours; options: the --option key=value pairs
        self.joint_names = ROBOT_JOINTS                   # the 62 action joints, in your training order

    def info(self):
        return {"joint_names": self.joint_names, "action_horizon": 16, "frame_hw": [360, 640],
                "camera_set": ["camera_ego_left", "camera_ego_right"], "action_representation": {"type": "absolute"}}

    def predict(self, images, state, prompt):
        # images: {"camera_ego_left": (H, W, 3) uint8 RGB, ...}; state: (62,) radians; prompt: str
        return self.policy(images, state, prompt)         # (action_horizon, 62) absolute radians
```

The 62 action joints are the names in the kit's `dexbench_policy_server/rby1_wuji2_joint_names.json`, which is the only
list of them. In that file's order:

| Indices | Joints |
| --- | --- |
| 0-6 | left arm (`left_arm_0` to `left_arm_6`) |
| 7-13 | right arm (`right_arm_0` to `right_arm_6`) |
| 14-15 | head (`head_0`, `head_1`) |
| 16-55 | both hands, 20 joints each, in blocks of five, thumb to little finger, left then right: 16-25 flexion (thumb CMC, finger MCP), 26-35 abduction (thumb CMC, finger MCP), 36-45 thumb MCP and finger PIP, 46-55 thumb IP and finger DIP |
| 56-61 | torso (`torso_0` to `torso_5`) |

If your model was trained in another order, state that order: `joint_names` may be any order of the same 62 names, and
the client maps them by name. An action is `(action_horizon, 62)` absolute joint positions in radians, in the order
`joint_names` states. The kit's [`example.py`](../../../source/dexbench_policy_server/dexbench_policy_server/example.py)
is a complete server that loads the file.

Then run it with one command, which is what the recipe's `command` starts. The server starts in the launcher's working
directory, not your code's, so the command changes to the directory that holds `my_module` first (or install your
module into that environment):

```bash
cd /path/to/your/code && exec dexbench-policy-server --policy my_module:MyPolicy --checkpoint {checkpoint} \
  --port {port} --token-file {token_file}
```

`dexbench_policy_server.example:HoldPose` is a complete example: it holds the robot still (it answers the state it
was given), which is enough to check the plumbing; `examples/own_policy_local.yaml` runs it. The kit implements the
contract the benchmark's HTTP client (`cosmos3_joint_policy.py`) and the launcher's worker expect:

- `GET /info` returns your `info()` (checked at start: a horizon below 16 or a missing key is refused before the ready
  line), with `server` added: the kit's name, version and contract version.
- `POST /predict` takes `{"images": {"camera_ego_left": "<base64 PNG>", ...}, "state": [62 floats],
  "prompt": "..."}` and returns `{"action": [[62 floats], ...]}`: `action_horizon` rows of finite absolute targets.
  Anything else your `predict` returns or raises comes back as `{"error": "..."}`, which stops the run with that
  message.
- Both refuse a request without `Authorization: Bearer <token>` (HTTP 401), the token in the file the launcher
  passes as `--token-file`.
- `GET /health` answers `{"status": "ok"}` without a token, for a readiness probe.
- Once `load` returns it prints `Server ready on 127.0.0.1:<port> ...`, flushed, which the recipe's default `ready`
  matches.
- One `predict` at a time: several simulators sharing the server (`colocated+N`) never run your model on two threads
  at once; their requests wait their turn.

A server of your own that speaks the same contract works too: the kit is the common way, not a requirement.

If your model was trained on arm-relative actions, add the measured state back before answering. The
benchmark always receives absolute targets.

## Make your server easy to run

The launcher starts your server on a fresh pod from the recipe: no CUDA toolkit, often no route to the Hugging Face
Hub, nothing of the machine you trained on. Each point below is something one of the benchmark's own models lacked,
and each one cost a failed launch before it was fixed. A server built on the kit meets 6 and 7 by construction.

1. **One named version of the server code.** Serve from a commit on a branch others can fetch, the one the
   checkpoint was trained with, and record it next to the checkpoint. A checkout nobody can name can't be reproduced.
2. **An environment built for that version.** Install it from the lockfile at that commit, on the serving machine.
   Running new code in an older commit's venv works only while the two lockfiles agree, and nothing says when they
   stop.
3. **A checkpoint that carries what it needs.** Nothing in its config should point at the training machine (`/data/...`,
   `/tmp/...`). Ship a dependency with the checkpoint, or name it by its Hub id and a pinned revision.
4. **No CUDA toolkit assumed.** Pods have none. If a library looks for one (Transformer Engine does), point it at the
   CUDA packages in your own environment.
5. **Works offline.** With `HF_HUB_OFFLINE=1` everything must load from the Hub cache, including from gated repos the
   pod has no token for. Some libraries still call the Hub by repo id; load from the cached directory instead.
6. **A token.** Take a token file (`--token-file` or `--api-token-file`) and refuse any request without it.
7. **A ready line, flushed.** Print one fixed line once the server answers requests, with `flush=True`: the worker
   waits for it in the server's log, and buffered output looks like a hung server.
8. **Shares the GPU.** Allocate memory as you use it. JAX reserves most of the GPU by default; set
   `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
9. **An interpreter the eval image has.** The server runs in the simulator's image, not the one its environment was
   built in. A uv venv links the build image's system Python when one matches (`/usr/bin/python3.10`), and the server
   then exits 127 in the eval image; build it with `UV_PYTHON_PREFERENCE=only-managed` so the venv uses an
   interpreter under uv's own directory, kept on the shared volume with the venv.


## 2. Register the model

Your eval config registers the model: no change to the repo. Give `model` a key of your own, a label and the
client's policy block, beside the checkpoint:

```yaml
model:
  key: mymodel                         # its name in the run manifests and the score; never a key models.py has
  label: My Model
  checkpoint: /path/to/your/checkpoint
  group: community                     # the default and the only group: reported apart, never headline
  policy:                              # the client's block: its class and parameters
    type: dexbench_arena_cosmos.policy.cosmos3_joint_policy.Cosmos3JointPolicy   # the HTTP client
    remote_url: http://127.0.0.1:5811  # the worker puts its server's port in it
    remote_api_token: ${oc.env:DEXBENCH_POLICY_API_TOKEN,null}   # the worker's token, from the environment
    policy_device: cpu
```

The launcher passes the block to every unit (`run_benchmark_cell.py --model-entry`), and each run manifest records
the whole entry (`model_entry`, with `registration: config`). `policy_variant` is filled in with the key and
`action_chunk_length` with the protocol's 16; a block that sets either to something else is refused, as is a key
`benchmark/models.py` already has. Leave the host at `127.0.0.1`, where the worker starts the server.

`--score` scores your model like any other, under the same protocol (horizon 16, the same rows and settings), and
reports it under `community` with its entry, apart from the headline models: it never makes their report partial, and
a run that departs from their settings is listed as a deviation of yours. Joining the headline table is a
maintainers' decision: a headline model is an entry in `source/dexbench_arena/dexbench_arena/benchmark/models.py`,
reviewed with the benchmark manifest it writes.

`policy.type` is any importable class: Arena imports it when a unit starts, as it does for the registered models. The
launcher checks neither the class nor its parameters before then, so step 3 is where a wrong one shows. Leave
`remote_api_token` out, or give it as `${oc.env:DEXBENCH_POLICY_API_TOKEN,null}`: the worker sets the token, and a
literal value would be written into the launch's records, so the launcher refuses one.

## 3. Check two episodes

Before a launch, run two episodes by hand on one GPU and watch their videos:

```bash
# inside the container, from the checkout (or a staged tree)
head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' > /tmp/token
(cd source/dexbench_policy_server && /isaac-sim/python.sh -m dexbench_policy_server --port 5811 --token-file /tmp/token \
    --policy dexbench_policy_server.example:HoldPose \
    --option cameras=camera_ego_left,camera_ego_right,camera_wrist_left,camera_wrist_right) &
export DEXBENCH_POLICY_API_TOKEN=$(cat /tmp/token)
/isaac-sim/python.sh scripts/evaluation/run_benchmark_cell.py --task r2 --cell train --model mymodel \
    --model-entry '{"key": "mymodel", "label": "My Model", "policy": {"type": "dexbench_arena_cosmos.policy.cosmos3_joint_policy.Cosmos3JointPolicy", "remote_url": "http://127.0.0.1:5811", "remote_api_token": "${oc.env:DEXBENCH_POLICY_API_TOKEN,null}", "policy_device": "cpu"}}' \
    --slices '{"tc1_mf3": [0, 2]}' --checkpoint <label> --output outputs/check-mymodel -- --viz none
```

The episodes' videos (one mp4 per camera) and results are under `outputs/check-mymodel/arena/`. If the robot
does something sensible, launch.

## 4. Launch every cell

Start from a copy of `examples/own_policy_local.yaml` in `outputs/configs/` (the simulator image mounts `outputs/`; its
own copy of the source does not see a config written later): set the `model` block (step 2), your checkpoint, and the
recipe's `command` to start *your* server in *your* environment. The worker fills in `{port}`, `{token_file}` and
`{checkpoint}`:

```yaml
servers:
  recipe:
    setup: test -e {checkpoint}              # every check that can fail before a unit: hashes, imports
    command: >-
      cd /path/to/your/repo && exec .venv/bin/python serve.py --checkpoint {checkpoint}
      --port {port} --token-file {token_file}
    ready: Server ready
```

Two rules for the recipe's text. The worker fills the fields in with Python's `str.format`, so a brace the shell
should see is doubled: `${{HOME}}`, not `${HOME}` (`--start` refuses a recipe that does not format). And each value
comes already shell-quoted: write `--checkpoint {checkpoint}`, not `"{checkpoint}"`, or a path with a space or a
quote in it reaches your server with the quotes inside it.

```bash
scripts/evaluation/eval_launch/launch.sh my_eval.yaml --local    # the driver places stations on this host's GPUs
scripts/evaluation/eval_launch/launch.sh my_eval.yaml --resume   # after a stop: the next driver recovers what the last
                                                                 # left, and nothing accepted reruns
```

On Kubernetes, generate the Job with `launch.sh my_eval.yaml --k8s-job <dir>` (`USER_GUIDE.md`, "On
Kubernetes"; `examples/kubernetes.yaml` and its site profile). The server then runs inside each worker pod, so its
environment has to be on the shared volume, built as item 9 above says. The
recipe's command runs in the pod's environment as it was before Isaac Sim's launcher changed it.

With `layout: colocated` the server shares its GPU with the simulator. GR00T, RLDX and pi0.5 fit next to one on a
46 GB L40S; Cosmos3 does not (its server takes 33 GB, and a simulator beside it ran out of memory on 2026-10-23), so it
runs `server+1`. `--smoke` measures what your server takes (USER_GUIDE.md, "Bringing your model onto Ray"). Check
that black or frozen frames don't appear in the videos: that is the sign of a GPU out of memory.

## 5. Score

```bash
scripts/evaluation/eval_launch/launch.sh my_eval.yaml --score
```

It writes `score_<task>.json` into the launch's output: each cell's success rate k/n with a 95% interval.
The scorer takes whole cells only, so it waits until every planned unit is done.

A cell scores 60 rows at `version: v1`, the default, and all 120 at `version: v1-plus` (`USER_GUIDE.md`).
Task titles and CLI keys are listed in the [evaluation protocol](../../../docs/Eval_Protocol.md).

| Task (website label) | Cells |
|---|---|
| Retail - Task 2, Retail - Task 3, Logistics - Task 1, Logistics - Task 2, Hotel - Task 1, Hotel - Task 2, Manufacturing - Task 5 | `train`, `ga_o` (new objects), `ga_s` (new scenes), `ga_os` (both) |
| Manufacturing - Task 1, Manufacturing - Task 2, Manufacturing - Task 3, Manufacturing - Task 4, Logistics - Task 3 | `train`, `ga_s` |
| Logistics - Task 4 | `train`, `ga_s`, `ga_p` (physics beyond training), `ga_sp` (scenes and physics beyond training) |
| Retail - Task 1 | `train`, `ga_o` |
| Hotel - Task 3 | `train`, `ga_os` |

## Your model already has its own server protocol

If you can't wrap the model in the HTTP contract (it already speaks ZMQ, websocket, …), write the client side
once. Everything that is the same for every model is in `RemoteJointPolicy`
(`dexbench_arena/policy/remote_joint_policy.py`): binding joints by name, 30 Hz scheduling, executing the
first 16 actions of each chunk, and recording what ran. You supply a `Transport` (how a request travels)
and a `Codec` (what goes in and comes out):

```python
@dataclass
class MyPolicyCfg(PolicyRateCfg):
    remote_url: str = "http://127.0.0.1:5811"   # or remote_host + remote_port: the launcher sets either
    remote_api_token: str | None = None
    action_chunk_length: int = 16
    policy_device: str = "cpu"
    robot_asset_name: str = "robot"
    joint_action_term: str = "joint_pos"
    policy_variant: str | None = None
    scheduler: str = "chunk"


class MyTransport(Transport):
    def info(self): ...                         # ask your server what it serves
    def infer(self, request, env_ids): ...      # send one request, return the answer


class MyCodec(Codec):
    requests = "each"                           # one request per environment ("all": one batched request)
    def check(self, info):                      # refuse a server that is not this model; then set:
        self.joint_names, self.action_horizon = info["joint_names"], info["action_horizon"]
    def bind(self, robot_joint_names):          # the robot's joint order, once
        self.state_ids = [robot_joint_names.index(n) for n in self.joint_names]
    def encode(self, observation, env_ids, task_description):
        # observation["camera_obs"]["robot_pov_left_cam_rgb"][env_id]: (360, 640, 3) uint8
        # observation["policy"]["robot_joint_pos"][env_id, self.state_ids]: the state in your order
        ...
    def decode(self, response, num_envs):       # -> (num_envs, action_horizon, len(joint_names)), absolute
        ...


@register_policy
class MyPolicy(RemoteJointPolicy, PolicyBase[MyPolicyCfg]):
    name = "my_policy"

    def __init__(self, config: MyPolicyCfg):
        super().__init__(config, MyTransport(config), MyCodec(config), config.action_chunk_length)
```

`cosmos3_joint_policy.py` is a complete example in about 200 lines (HTTP), and `gr00t_joint_policy.py`
uses ZMQ. Register it as in step 2, with `policy: {type: your_module.MyPolicy, remote_api_token:
${oc.env:DEXBENCH_POLICY_API_TOKEN,null}, ...}`, so the token is never written into a run's files. If the simulator needs your client package, list it in the config's `backend.pythonpath`.

## When something goes wrong

The server's log is `logs/<worker>/server-<port>-<worker start>-<n>.log`, and each unit's simulator log is
`units/<unit>/<attempt>.log`. `launch.sh my_eval.yaml --status` lists every failed unit with its reason.

| What you see | Why, and what to do |
|---|---|
| The worker stops with "policy server exited" or "setup failed" | Read the server log. The unit was handed back untried; fix the recipe and `--resume`. |
| `Live joint action contract differs from the model's joint names` | `/info`'s `joint_names` are not the robot's 62 action joints. The kit's `dexbench_policy_server/rby1_wuji2_joint_names.json` lists them. |
| `Unsupported Wuji2 camera keys` | `camera_set` names a camera that does not exist. Use the four keys above. |
| `Camera frames are 640x360, a different aspect ratio …` or `… smaller than the policy's …` | `frame_hw` is not 16:9, or is larger than 360x640. |
| `Cosmos3 server /info: HTTP 401` | Your server reads another token than the one the launcher wrote: read `--token-file`. |
| `Invalid Cosmos3 action chunk` | `/predict` answered another shape than `(action_horizon, 62)`, or a NaN. |
| `action_chunk_length must be in 1..N` | Your `action_horizon` is below 16. Answer at least 16 actions. |
| The robot moves, but wildly or too slowly | The training data's rate or action representation differs: check for 30 Hz, absolute radians, and the state added back for relative models. |
| Black or frozen camera images in the videos | The GPU is out of memory. Make your server lighter, or give it its own GPU. |

## Further reading

- `USER_GUIDE.md`: the launcher, resuming, and the config reference.
- `docs/Eval_Protocol.md`: the cells, the episode caps, how models are compared, and the scorer.


## Leaderboard submission evidence

For a submission launch, set `submission: true` and one `model.checkpoint` plus its
`model.checkpoint_sha256`. Workers enable the native trajectory identity recorder and
record the digest separately from the checkpoint path. Use a clean evaluator checkout.
Keep the same config-model block as `model_entry.json` in the submission bundle; its
policy group stays `community`. Maintainer registration in the separate leaderboard
repository determines publication, without adding a required headline baseline.

Complete every release task and cell. Only completed, non-overlapping unit outputs
are automatically submittable. Output that depends on the launcher's accepted-row
selection for retry/salvage is not yet supported by the submission validator. See the
[submission guide](../../../docs/Leaderboard_Submission.md) for the evidence contract.
