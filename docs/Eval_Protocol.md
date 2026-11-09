# DexBench evaluation protocol

This document tells you how DexBench scores a policy. It gives the tasks and
their cells, the files that make up a cell, how to serve a policy, how to run a
cell and how the scorer aggregates the results.

The protocol is code, in the package `dexbench_arena.benchmark`
(`source/dexbench_arena/dexbench_arena/benchmark/`):

- `protocol.py` holds the fixed numbers: cells, rows, versions, caps, policy rate.
- `models.py` holds the models and their policy blocks.
- `spec.py` holds one declaration per task: objects, ranges, cap source, sweep.

If this document and the code disagree, the code is correct.

Task labels and titles match the [DexBench website](https://dexbench.ai/#tasks).
Use the CLI keys in the task table below for commands and dataset paths.

For installation, assets and datasets, see [Getting started](Getting_Started.md). For Docker,
see [docker/README.md](../docker/README.md).
Run every command from the root of the checkout. Run each `/isaac-sim/python.sh`
command in the container.

## Terms

| Term | Meaning |
| --- | --- |
| Task | One benchmark task, for example Hotel - Task 1. Its key (`h1`) is the `--task` value. |
| Cell | One evaluation condition of a task: an object split and a spatial range (§2). |
| Row | One frozen start state. Each row gives one episode. |
| Run | One entry under `runs:` in a cell file: one object at one spatial setting. It replays its own row file. |
| Frozen set | The checked-in list of row files, with the SHA-256 and the row count of each file. |
| Version | The rows of a cell that a model runs: `v1` (60 rows) or `v1-plus` (120 rows). |
| Model | One entry in `benchmark/models.py`. Its key (`gr00t`) is the `--model` value. |
| Run directory | The `--output` directory of one model run: one model on one cell, or on one sweep. |
| Cap | The maximum duration of an episode, in seconds. |
| Sweep | The sensitivity sweep of a task (§11). It never changes the score. |

## 1. What the benchmark measures

DexBench measures dexterous manipulation by one robot: the RBY1 with two Wuji
hands (embodiment `dexbench_rby1_wuji2_joint`). A policy commands 62 joint
position targets: both arms, both hands, the torso and the head.

Each task compares the success rate in the training conditions with the success
rate on held-out objects and on placements beyond the training range.

| Task | CLI key | Environment task | Cells | Episodes per model, v1 / v1-plus | Cap |
| --- | --- | --- | --- | --- | --- |
| Retail - Task 1<br>Diverse Fruit Picking | `r1` | retail, `pick_fruit_from_display` | Train, GA-O | 120 / 240 | 50 s |
| Retail - Task 2<br>Dense-Clutter Selective Retrieval | `r2` | retail, `retrieve_beverage_from_clutter` | Train, GA-O, GA-S, GA-OS | 240 / 480 | 35 s |
| Retail - Task 3<br>Spatially-Constrained Slot Placement | `r3` | retail, `place_box_in_slot` | Train, GA-O, GA-S, GA-OS | 240 / 480 | 75 s |
| Logistics - Task 1<br>Conveyor Pick | `l1` | logistics, `pick_polybag_from_conveyor` | Train, GA-O, GA-S, GA-OS | 240 / 480 | 60 s |
| Logistics - Task 2<br>Parcel Cage Unloading | `l2` | logistics, `lift_to_conveyor` | Train, GA-O, GA-S, GA-OS | 240 / 480 | 25 s |
| Logistics - Task 3<br>Paper Insert Kitting | `l3` | logistics, `pack_invoice` | Train, GA-S | 120 / 240 | 25 s |
| Logistics - Task 4<br>Poly Bag / Parcel Flip Induction | `l4` | logistics, `deliver_parcels_barcode_up` | Train, GA-S, GA-P, GA-SP | 240 / 480 | 95 s |
| Manufacturing - Task 1<br>Precision Seating / Mechanical Insertion | `m1` | manufacturing, tabletop `pick_insert_drawer` | Train, GA-S | 120 / 240 | 100 s |
| Manufacturing - Task 2<br>Connector Insertion | `m2` | manufacturing, `insert_connector` | Train, GA-S | 120 / 240 | 70 s |
| Manufacturing - Task 3<br>Power-Driver Screw Fastening | `m3` | manufacturing, `power_driver_fastening` | Train, GA-S | 120 / 240 | 70 s |
| Manufacturing - Task 4<br>Bolt Hand-Threading | `m4` | manufacturing, `hand_start_leg` | Train, GA-S | 120 / 240 | 115 s |
| Manufacturing - Task 5<br>Moving-Conveyor Kitting | `m5` | manufacturing, `place_part_in_kit_box` | Train, GA-O, GA-S, GA-OS | 240 / 480 | 30 s |
| Hotel - Task 1<br>Stable Stacking / Nesting | `h1` | banquet hall, `stack_plates` | Train, GA-O, GA-S, GA-OS | 240 / 480 | 55 s |
| Hotel - Task 2<br>Liquid Pour & Serve | `h2` | banquet hall, `lift_and_pour` | Train, GA-O, GA-S, GA-OS | 240 / 480 | 90 s |
| Hotel - Task 3<br>Bimanual Singulation | `h3` | banquet hall, `extract_utensil` | Train, GA-OS | 120 / 240 | 25 s |

**Score.** For each cell, the score is the success rate k/n with its 95% interval
(§10). For each task, the score is also the macro-average: the unweighted mean of
the cell success rates. The scorer does not combine tasks.

## 2. Cells

A cell is an object split and a spatial range. A task with a physical axis also
has a physical range.

| Cell | `--cell` | Objects | Spatial range | Physical range |
| --- | --- | --- | --- | --- |
| Train | `train` | training | training | training |
| GA-O | `ga_o` | held out | training | training |
| GA-S | `ga_s` | training | beyond training | training |
| GA-OS | `ga_os` | held out | beyond training | training |
| GA-P | `ga_p` | training | training | beyond training |
| GA-SP | `ga_sp` | training | beyond training | beyond training |

A task has only the cells that its axes support:

- Logistics - Task 3, Manufacturing - Task 1, Manufacturing - Task 2, Manufacturing - Task 3 and Manufacturing - Task 4 have one object and no held-out object.
  They have Train and GA-S.
- Retail - Task 1 declares no spatial range beyond its training layouts. It has Train and GA-O.
- Logistics - Task 4 has no held-out objects: its 12 parcel sequences use training parcels.
  Its physical axis, the parcel mass, adds GA-P and GA-SP.
- Hotel - Task 3 has Train and GA-OS. Its GA-OS couples held-out utensil trios with the
  unseen grasps that their geometry gives, made by the training procedure. Its pose
  ranges can overlap training. Thus it is not a separate test of spatial range.

**Rows and Runs.** Each cell holds 120 frozen rows. The rows are divided over the
Runs of the cell. A range drawn per episode gives one Run per object. A range of
fixed settings gives one Run per object and setting, for example "the base 4 cm
out toward the east". Each Run draws at its own seed. In Logistics - Task 2 and Logistics - Task 3, the
Runs of a cell share one seed, so some draws repeat across Runs row by row. The
rows of a cell are distinct start states. In Hotel - Task 3 they are distinct pairs of
start state and target.

No row was selected on the outcome of a policy. Most rows were recorded under a
policy that holds the reset pose, and kept only when the scene came to rest. The
other rows were written from frozen draws and checked as valid starts. Every
model replays the same rows.

**Objects.** The number in parentheses is the rows per object in each cell of that split.

| Task | Training objects (Train, GA-S, GA-P, GA-SP) | Held-out objects (GA-O, GA-OS) |
| --- | --- | --- |
| Retail - Task 1<br>Diverse Fruit Picking | plum, peach, apple, pear piles (30) | lime, mandarin, lemon, bell pepper piles (30) |
| Retail - Task 2<br>Dense-Clutter Selective Retrieval | drinks TC1, TC2, TP1 (40), in lanes MF3 to MF5 | EC1, EC2, EP2, EP3 (30), in lanes MF3 to MF5 |
| Retail - Task 3<br>Spatially-Constrained Slot Placement | cartons R3-T1 to R3-T4 (30) | R3-E1, R3-E2, R3-E4, R3-E5, R3-E6 (24) |
| Logistics - Task 1<br>Conveyor Pick | bag sets `train_a` to `train_d` (30) | `eval_a`, `eval_b` (new sizes), `eval_c`, `eval_d` (new colours) (30) |
| Logistics - Task 2<br>Parcel Cage Unloading | boxes `train_0`, `train_1` (60) | `eval_0` to `eval_3` (30) |
| Logistics - Task 3<br>Paper Insert Kitting | one invoice sheet, `invoice_202x276` (120) | none |
| Logistics - Task 4<br>Poly Bag / Parcel Flip Induction | parcel sequences `eval_00` to `eval_11` (10) | none |
| Manufacturing - Task 1<br>Precision Seating / Mechanical Insertion | one drawer, `lg_fh25eae_detergent_drawer` (120) | none |
| Manufacturing - Task 2<br>Connector Insertion | one connector, `S2` (120) | none |
| Manufacturing - Task 3<br>Power-Driver Screw Fastening | one station, `center_hinge_fixture` (120, in 8 builds of 15) | none |
| Manufacturing - Task 4<br>Bolt Hand-Threading | one leg, `lg_fx25_leveling_leg` (120) | none |
| Manufacturing - Task 5<br>Moving-Conveyor Kitting | M24 bolt, 6206 bearing, 50A elbow, POM roller, NEMA 23 stepper (24) | 10 parts, each the shape of a training part at another size (12) |
| Hotel - Task 1<br>Stable Stacking / Nesting | `harmynta_deep_plate_22`, `ikea_365_bowl_16`, `ikea_365_bowl_22` (40) | `godmiddag_deep_plate_23`, `godmiddag_bowl_16`, `ikea_365_bowl_19` (40) |
| Hotel - Task 2<br>Liquid Pour & Serve | jugs `idealisk`, `d00527_x15` (60) | `steel_pitcher_large`, `steel_pitcher_footed` (60) |
| Hotel - Task 3<br>Bimanual Singulation | 4 training utensil trios, each with a spoon, fork or knife target (10) | 4 held-out trios, same targets (10) |

**Spatial ranges.** Each task's `SpatialRange.note` in `spec.py` gives the full
numbers and reasons. Each cell file repeats them in its header.

| Task | Training range | Beyond training |
| --- | --- | --- |
| Retail - Task 1<br>Diverse Fruit Picking | the pile's start layout, from the kind's evaluation layouts; base fixed | none |
| Retail - Task 2<br>Dense-Clutter Selective Retrieval | target in lane MF3, MF4 or MF5; base x, y ±2 cm, turned ±3° | base fixed 4 cm out toward e, w, sw, s or se, turned −0.5° to +3°; or base turned +6° or −4° at its training x, y |
| Retail - Task 3<br>Spatially-Constrained Slot Placement | base x, y ±2 cm, turned −2.5° to +3°; carton in the tote x, y ±2 cm | one axis fixed out, the other at its training draw: base 4.5 cm out in one of 8 directions, turned −1° to +3°; or carton 4 cm out in one of 8 directions; or base turned +6° or −5.8° |
| Logistics - Task 1<br>Conveyor Pick | each bag's long side 0° to 30° off the belt direction | each bag's long side 60° to 90° off the belt direction |
| Logistics - Task 2<br>Parcel Cage Unloading | cage stack depth 0 to 80 mm, lateral ±10 mm | depth −50 to +88 mm, lateral ±13 mm, outside the training box |
| Logistics - Task 3<br>Paper Insert Kitting | 10 axes (sheet pile, box, bench height, base), each in its training interval | one axis fixed at 1.5 times its training extent on one side (20 settings) |
| Logistics - Task 4<br>Poly Bag / Parcel Flip Induction | each parcel put down at an in-domain rotation, square to the belt | each rotation turned 45° about up |
| Manufacturing - Task 1<br>Precision Seating / Mechanical Insertion | base x, y ±3 cm; drawer and washer x, y ±5 cm; all turned ±5°; full support | the supported part of 1.5 times that envelope, outside training on one coordinate or more |
| Manufacturing - Task 2<br>Connector Insertion | plug tape spot in a 4 mm disk, turned −10° to +20°; base ±10 mm, ±1° | spot in the 4 to 6 mm ring, turned +25° to +30°; base ±12.5 mm, ±1.25° |
| Manufacturing - Task 3<br>Power-Driver Screw Fastening | base, driver and station each x, y ±10 mm | base fixed 30 mm out in one of 8 directions |
| Manufacturing - Task 4<br>Bolt Hand-Threading | base ±20 mm, leg ±5 mm, cabinet ±20 mm, in x and y | leg fixed 15 mm out in one of 8 directions; or cabinet at x ±30 mm, y ±28 mm or a corner |
| Manufacturing - Task 5<br>Moving-Conveyor Kitting | empty kit-box compartment: far-right or near-left | empty compartment: near-right or far-left |
| Hotel - Task 1<br>Stable Stacking / Nesting | base and both stacks each in a 0 to 3 cm disk; base turned ±5° | each in a 3 to 5 cm ring; base turned ±5° |
| Hotel - Task 2<br>Liquid Pour & Serve | jug x, y ±2 cm, turned ±10°; goblet ±1 cm; base x, y ±1 cm, turned ±5° | one axis fixed out, the others at their training draws: jug 4 cm out in one of 8 directions (3 cm toward w and s); jug turned ±15°; base 2 cm out in one of 8 directions; base turned ±7°; goblet 2 cm out in one of 8 directions |
| Hotel - Task 3<br>Bimanual Singulation | saved in-hand grasps of the training trios, none a start of a training demo | held-out trios with unseen grasps (see above) |

**Physical axis (Logistics - Task 4).** In the training range, each parcel's mass is its
nominal mass times ×0.5, ×0.71, ×1, ×1.41 or ×2, drawn per parcel at each reset.
Beyond training, the multiplier is ×0.25, ×0.35, ×2.83 or ×4.

**Small beyond steps.** Some beyond ranges are a narrow step past training. A
small gap between Train and GA-S is expected there:

- Logistics - Task 2: most beyond rows put the stack up to 50 mm nearer the robot. The deeper
  and lateral margins are at most 8 mm and 3 mm.
- Manufacturing - Task 2: the yaw step is one-sided, past a held-out gap of +20° to +25°.
- Manufacturing - Task 3: only the base moves out, 20 mm past its training square on an axis.
- Manufacturing - Task 4: the cabinet is 10 mm (x) and 8 mm (y) past training; the leg 8 to 10 mm.
- Manufacturing - Task 5: a GA-S row differs from a Train row in the empty compartment only.

**Retail - Task 2 held-out cans.** In scenes with a training drink as target, the
delivered stock keeps EC1 cans in lane MF2 and EC2 cans in lane MF5. With the
target in MF3 or MF4, some of these cans stand next to it. A policy trained in
these scenes has seen EC1 and EC2. Thus read GA-O and GA-OS per object: EC1 and
EC2 apart from EP2 and EP3. The report gives each object's k/n (§10).

**Physics.** Every cell runs on the Newton physics backend with the task's
solver profile. The Logistics - Task 2 and Hotel - Task 2 cells use 8 physics substeps.

## 3. Versions

The benchmark has two versions over the same frozen rows:

- `v1` (the default) runs a fixed 60 of the 120 rows of each cell.
- `v1-plus` runs all 120 rows.

The v1 rows are split over the objects of the cell in proportion to their rows,
then over the Runs of each object. Each Run keeps evenly spaced pairs of
consecutive rows. The v1 rows are pinned in
`benchmark/frozen/versions_db_<task>.json`. The runner and the scorer read them
from that file.

A `v1-plus` run also gives a `v1` score: `score --version v1` counts its v1 rows
only. Use `v1-plus` when `v1` cannot separate two models (§13).

## 4. Files that make up a cell

| File | Content |
| --- | --- |
| `scripts/evaluation/experiment_configs/benchmark/db_<task>_eval_<cell>.yaml` | The cell. One Run per object and spatial setting. Each Run replays its row file, one episode per row, at the cap. Each Run holds the baseline policy, `HoldPosePolicy`. |
| `scripts/evaluation/experiment_configs/benchmark/db_<task>_sensitivity.yaml` | The sweep. One Run that replays the 20 build files of the sweep. |
| `scripts/evaluation/experiment_configs/benchmark/benchmark_manifest.json` | For each task: the cap, the cells, each Run's object, row file, row count and seed, each cell's approved settings (`declared`), the policy rate, the sweep and the registered models. |
| `source/dexbench_arena/dexbench_arena/benchmark/frozen/db_<task>.json` | The frozen set of the cells. |
| `source/dexbench_arena/dexbench_arena/benchmark/frozen/db_<task>_sensitivity.json` | The frozen set of the sweep, kept apart. |
| `source/dexbench_arena/dexbench_arena/benchmark/frozen/versions_db_<task>.json` | The v1 rows of each row file. |
| `outputs/db_conditions/` | The rows: one `<Run>.jsonl` per Run. The manifest gives each path. |

No cell file names a model. If you run a cell file as it is, the baseline
policy runs. That run is a pipeline check, and the scorer refuses it.

Get the rows on the host, in the checkout:

```bash
./scripts/download_datasets.sh --eval-conditions
```

The script downloads the rows from
[`RLWRLD/DexBench-eval`](https://huggingface.co/datasets/RLWRLD/DexBench-eval) at the
revision that `reproducibility/datasets.lock.json` pins. The same download installs the Retail - Task 1
layouts and the Manufacturing - Task 5 parts-bin piles under `outputs/db_conditions/`, and Hotel - Task 3's
runtime inputs in `outputs/db_h3_inputs/`. Hotel - Task 3's utensils are asset packages
under `assets/`, not part of this download.

Before the script installs anything in `outputs/`, it checks every row file
against the frozen sets. It also checks that Hotel - Task 3's inputs give Hotel - Task 3's frozen
rows. The check needs only `python3`. The script refuses to overwrite a local file
that differs; `--force` replaces it.

## 5. Serve a policy

**Cameras.** Every cell renders four cameras at 640×360 (`camera_scale: 0.5`, the
training render):

- head cameras `robot_pov_left_cam_rgb`, `robot_pov_right_cam_rgb`;
- palm cameras `wrist_left_cam_rgb`, `wrist_right_cam_rgb`.

Each policy server declares the cameras that it reads. The client stops the run
if the server declares a camera that the cell does not render.

**Instruction.** Every model gets the environment's own task description as its
language instruction.

**Registered models.** Each model runs next to the simulator as a policy server.
The client in the policy block connects to it.

| Key | Group | Client class (package) | Default address | Protocol |
| --- | --- | --- | --- | --- |
| `gr00t` | headline | `Gr00tJointPolicy` (`dexbench_arena_gr00t`) | `127.0.0.1:5810` | GR00T ZMQ |
| `cosmos` | headline | `Cosmos3JointPolicy` (`dexbench_arena_cosmos`) | `http://127.0.0.1:5811` | HTTP |
| `rldx11` | headline | `RldxRby1WujiJointRemotePolicy` (`dexbench_arena_rldx`) | `127.0.0.1:5813` | RLDX ZMQ |
| `pi05` | headline | `Pi0Rby1WujiJointRemotePolicy` (`dexbench_arena_openpi`) | `127.0.0.1:5814` | openpi websocket |
| `fastwam` | headline | `WamJointPolicy`, `model_family: fastwam` (`dexbench_arena_wam`) | `http://127.0.0.1:5815` | HTTP |
| `openwam` | headline | `WamJointPolicy`, `model_family: openwam` (`dexbench_arena_wam`) | `http://127.0.0.1:5816` | HTTP |
| `gr00t_rel` | ablation | as `gr00t`, both arms relative to the state | as `gr00t` | GR00T ZMQ |
| `cosmos_rel` | ablation | as `cosmos`, both arms relative to the state | as `cosmos` | HTTP |
| `astra_l0`, `astra_l1_generic`, `astra_l1_shapes` | trial | `VlmGoalPolicy` (`dexbench_arena_vlm_agent`) | OpenAI Responses API | HTTPS |

The `note` of each entry in `models.py` gives the training recipe and how the
server is started. Two models need code that is not public yet: the `rldx11`
client imports `rldx.policy.server_client` from RLWRLD/RLDX, and the `pi05`
server is `scripts/serve_policy.py` of RLWRLD/openpi. The reference checkpoints will be released on Hugging Face
([Reference checkpoints](Getting_Started.md#reference-checkpoints)). The server-side settings are fixed
per model, the same for every cell:

- `gr00t`: 4 denoising steps; the client area-resizes each image to 320×180.
- `cosmos`: 10 UniPC denoising steps, shift 5. Start the server with
  `--num-steps 10 --shift 5 --sampler unipc`; its default is 4 steps.
- `rldx11`: 10 uniform denoising steps (timesteps 0, 0.1, ..., 0.9), stateless.
  The server does not state its timesteps, so the Run declares them and each row
  records them. The scorer refuses rows that record no timesteps. It scores rows
  that record other timesteps, and lists them as a deviation (§10, Status). The
  client sends the raw 640×360 images.
- `pi05`: 10 denoising steps. The client letterboxes each image to 224×224.
- `fastwam`: 10 denoising steps, flow shifts 5 (video) and 1 (action), noise seed 42.
- `openwam`: 10 denoising steps.

The HTTP servers state their joint names, cameras, frame size and action horizon
at `GET /info`. The client refuses a server whose statement does not agree with
the Run. For example, `fastwam` refuses a server that states another
`model_family`, and `cosmos` refuses a server that serves relative actions.

**GR00T joint contract.** The GR00T client reads a contract from
`/tmp/dexbench_gr00t_n17_contract/policy.yaml`. Write it in the container. Give
the video keys of your checkpoint, in its order:

```bash
/isaac-sim/python.sh -m dexbench_arena_gr00t.policy.wuji2_gr00t_config \
    --output /tmp/dexbench_gr00t_n17_contract \
    --cameras camera_ego_left camera_wrist_left camera_wrist_right
```

Without `--cameras`, the tool writes the two head cameras. The client refuses a
server whose video keys differ from the contract.

**Token.** If your server requires a token, set `DEXBENCH_POLICY_API_TOKEN` in the
environment of `run_benchmark_cell.py`. The `pi05` client sends no token. No file
that a run writes holds the token.

**Groups.** The scorer reports each group apart:

- `headline`: required. A missing headline model makes the report `partial`
  (§10, Status).
- `ablation`: an action-representation study of a headline model. It runs every
  cell from its own checkpoint.
- `trial`: a planner that chooses its own re-plan points. The Astra trials run
  only the Train cell of Retail - Task 2, Logistics - Task 2, Manufacturing - Task 1 and Hotel - Task 1, on 10 (`astra_l0`) or 20
  (`astra_l1_*`) rows: the first v1 rows of each Run. They need `OPENAI_API_KEY`, and
  `--set cost_budget_usd=... --set cost_ledger_path=...`.

### Evaluate your own model

1. Serve the model. Use one of these two methods:
   - Write a server that speaks the HTTP protocol of `Cosmos3JointPolicy`.
     `GET /info` returns `joint_names` (the 62 joint names of
     `RBY1_WUJI2_JOINT_ACTION_NAMES` in `dexbench_arena.embodiments.rby1_wuji2.constants`,
     in your order), `frame_hw`, `action_horizon` (16 or more) and `camera_set` (a
     list such as `["camera_ego_left", "camera_wrist_left", "camera_wrist_right"]`).
     `POST /predict` receives `images` (base64 PNG per camera key), `state` (the
     joint positions in your order) and `prompt`. It returns `action`: absolute
     joint targets, shape `[action_horizon, 62]`. If a token is set, each
     request carries `Authorization: Bearer <token>`.
   - Write a client class on `RemoteJointPolicy`
     (`dexbench_arena/policy/remote_joint_policy.py`). Give it a `Transport` (how
     requests reach your server) and a `Codec` (the request, the action chunk and
     the contract check). Base its config on `PolicyRateCfg`. The five clients
     above are examples.
2. Add a `ModelEntry` to `MODELS` in `benchmark/models.py`. Set `key`, `label`,
   `note` and a policy block with `type` (the dotted class path),
   `policy_variant` (equal to the key), `action_chunk_length: 16`,
   `policy_device` and the server address.
3. Add the key to `models` of each task in `benchmark_manifest.json`:
   `"<key>": {"label": "<label>", "available": true, "group": "headline"}`.
   The runner reads the model from `models.py`. The scorer reads the list of
   models from the manifest.

A report that does not include all headline models has status `partial`. Your
model's own line is complete when it has every cell of the task (§10, Status).

Do not use `scripts/evaluation/env_server.py` for a score. It serves an
environment without the frozen rows and, by default, without the task's
terminations.

## 6. Run a model on a cell

Start the policy server. Then, in the container:

```bash
/isaac-sim/python.sh scripts/evaluation/run_benchmark_cell.py --task h1 --cell train --model gr00t \
    --output outputs/db_h1_eval_train_gr00t --checkpoint <label of the loaded checkpoint> -- --viz none
```

Run each cell of the task into its own run directory.

| Option | Use |
| --- | --- |
| `--task`, `--cell` | The task key and the cell. `--sweep` replaces `--cell` for the sweep (§11). |
| `--model` | A key of `benchmark/models.py`. |
| `--output` | The run directory. One directory holds one run. |
| `--checkpoint` | A label for what the server loaded. The runner records it and loads nothing. The scorer requires it, and requires one label per model. |
| `--set FIELD=VALUE` | Changes a deployment field of the policy block: `remote_host`, `remote_port`, `remote_url`, `policy_config_yaml_path`, `policy_device`, `cost_budget_usd`, `cost_ledger_path`. Example: `--set remote_port=5812`. |
| `--version` | `v1` (default) or `v1-plus`. |
| `--shard I/N` | Runs part I of N (I from 0) of every Run's rows. Run each part into its own directory. |
| `--slices JSON` | Runs exactly the given rows of the given Runs: `{"<Run>": [start, end]}`, end excluded. `start` and `end` are positions in the Run's rows at `--version`, not frozen row numbers. |
| `--prepare_only` | Checks and writes the composed file and the run manifest. It does not run. |
| `--no_record_dataset`, `--no_record_camera_video` | Do not keep the per-step HDF5, or the camera videos. |
| `-- ...` | Arguments after `--` go to the experiment runner, for example `--viz none`. |

Before the simulator starts, the runner puts the model's policy block, the four
cameras and the policy rate (§7) in each Run. It refuses the run when:

- the frozen set is missing, or a row file is not in it or changed;
- the composed Runs do not build the approved settings of the cell;
- `<output>/arena` exists;
- one of these variables is set: `DEXBENCH_RENDER_INTERVAL`, `DEXBENCH_DLSS_MODE`,
  `DEXBENCH_WATER` and every `DEXBENCH_WATER_*`, `DEXBENCH_HIDE_PRIMS`,
  `DEXBENCH_RENDER_TRANSLUCENCY`, `DEXBENCH_RENDER_REFLECTIONS`,
  `DEXBENCH_RECORD_EXPORT_ON_RESET`.

Then it runs `scripts/evaluation/experiment_runner.py` on the composed file. The
run directory then holds:

| Path | Content |
| --- | --- |
| `composed_experiment.yaml` | The cell file with the model's policy block. |
| `run_manifest.json` | The model, its group, policy block, horizon, deployment fields, checkpoint label, version and row indices; the code revision; the SHA-256 of the cell file, the composed file, the manifest and the frozen set. |
| `rows/<Run>.jsonl` | The exact rows of a Run that replays only part of its rows. |
| `arena/` | The experiment output: `arena_experiment_result.json`; per Run, the episode results (`episode_results*.jsonl`), an mp4 per camera and the per-step HDF5 (actions, policy observations, scene states, no camera frames, about 3.2 KB per step). The client writes the server's statement to `policy_provenance.json`. |

The recording options do not change the score.

## 7. Policy rate and execution horizon

**Execution horizon.** Every headline model and ablation executes 16 actions of
each chunk (`action_chunk_length: 16`). Then the client asks the server again. A
trial has no chunk and no horizon.

**Policy rate.** The cell files state a 30 Hz control step. A model run changes
this:

- The control step runs at 60 Hz (`rby1_control_hz: 60`).
- The client asks the policy at 30 Hz (`policy_hz: 30`), on the observation of
  that step. The chunk advances one action per policy step.
- The control step on which a new action arrives commands the midpoint between
  the previous action and the new action. The next step commands the new
  action (`policy_action_interpolation: linear`).
- The first action after a reset is held for both steps.
- The cameras render a new frame only on the policy steps (`camera_hz: 30`).
- One process runs one environment (`num_envs: 1`).

A trial keeps the 30 Hz control step, and the client asks the policy at every step.
The cap stays in seconds. The per-step HDF5 has 60 Hz rows, and the camera videos
repeat each policy frame. Each policy stamps its model key, horizon and rate in
the episode results. The scorer holds each episode to the rate of the task.

## 8. Episode caps

One rule gives the cap of every task, from its 240 training demos:

```text
cap = max(2 × median, 1.25 × longest) + hold, rounded up to the next 5 s
```

The hold adds time for a success dwell that the demos do not contain. `spec.py`
sets it to 1 s for Retail - Task 2 and Logistics - Task 3, and to 0 s for all other tasks.

| Task | Median | Longest | Hold | Cap |
| --- | --- | --- | --- | --- |
| Retail - Task 1<br>Diverse Fruit Picking | 6.5 s | 37.2 s | 0 s | 50 s |
| Retail - Task 2<br>Dense-Clutter Selective Retrieval | 16.6 s | 22.7 s | 1 s | 35 s |
| Retail - Task 3<br>Spatially-Constrained Slot Placement | 35.6 s | 55.9 s | 0 s | 75 s |
| Logistics - Task 1<br>Conveyor Pick | 25.5 s | 44.7 s | 0 s | 60 s |
| Logistics - Task 2<br>Parcel Cage Unloading | 10.8 s | 12.8 s | 0 s | 25 s |
| Logistics - Task 3<br>Paper Insert Kitting | 8.5 s | 18.6 s | 1 s | 25 s |
| Logistics - Task 4<br>Poly Bag / Parcel Flip Induction | 42.0 s | 74.9 s | 0 s | 95 s |
| Manufacturing - Task 1<br>Precision Seating / Mechanical Insertion | 48.7 s | 61.1 s | 0 s | 100 s |
| Manufacturing - Task 2<br>Connector Insertion | 33.4 s | 36.8 s | 0 s | 70 s |
| Manufacturing - Task 3<br>Power-Driver Screw Fastening | 34.5 s | 35.6 s | 0 s | 70 s |
| Manufacturing - Task 4<br>Bolt Hand-Threading | 56.2 s | 60.0 s | 0 s | 115 s |
| Manufacturing - Task 5<br>Moving-Conveyor Kitting | 14.4 s | 20.0 s | 0 s | 30 s |
| Hotel - Task 1<br>Stable Stacking / Nesting | 25.9 s | 29.7 s | 0 s | 55 s |
| Hotel - Task 2<br>Liquid Pour & Serve | 43.2 s | 49.5 s | 0 s | 90 s |
| Hotel - Task 3<br>Bimanual Singulation | 5.4 s | 16.5 s | 0 s | 25 s |

`DemoDurations.source` in `spec.py` names each training set. An episode ends at
success, at a failure termination or at the cap (Logistics - Task 1: see §9). An episode that
reaches the cap without success is a failure.

## 9. Success criteria

Success is the task's success term at the current step. All conditions in one
line of the table must hold together, for the stated time.

| Task | Success | Early failure |
| --- | --- | --- |
| Retail - Task 1<br>Diverse Fruit Picking | A fruit of the instructed kind is in the basket and at rest. The hands carry at most min(1 N, 25% of its weight). All other fruit stay on the tray. For 1.0 s. | A fruit leaves the workspace or drops; a fruit at rest off the tray and the basket; a wrong kind or two fruits in the basket. |
| Retail - Task 2<br>Dense-Clutter Selective Retrieval | The target's centre is in the basket, the target moves slower than 0.10 m/s and no watched neighbour is knocked. On 90% of a sliding 1.0 s window. | The target drops or leaves the workspace; a wrong pick; a neighbour topples. |
| Retail - Task 3<br>Spatially-Constrained Slot Placement | The carton is in the empty slot (±20 mm), upright and at rest. The hand force on it is at most 1 N, the wrist is 5 cm or more in front of it, and no neighbour is disturbed. For 1.0 s. | The carton drops or leaves the workspace; a neighbour topples or falls. |
| Logistics - Task 1<br>Conveyor Pick | Headline: the first two bags in stream order are placed before any failure. A bag is placed when 50% or more of its particles are in the tote, at rest, for 0.5 s. | A bag drops or is missed; a box is disturbed. |
| Logistics - Task 2<br>Parcel Cage Unloading | The box is on the conveyor deck, not on a long side (tilt at most 20°), at rest. For 0.5 s. | The box drops, falls fast or leaves the workspace. |
| Logistics - Task 3<br>Paper Insert Kitting | A sheet is in the carton: its centroid in, 90% or more of its particles in, maximum particle speed below 0.30 m/s. No dwell. | A sheet falls 0.15 m below its pile. |
| Logistics - Task 4<br>Poly Bag / Parcel Flip Induction | All five parcels are delivered. A parcel is delivered when it rests on FB1 at the judging line for 0.1 s, carried at 50% or more of belt speed, barcode within 60° of up. | The first failed parcel ends the episode: barcode more than 60° off up, a fall, or not judged within 20 s. |
| Manufacturing - Task 1<br>Precision Seating / Mechanical Insertion | The drawer is seated: −1 to +2 mm in depth, within 2 mm laterally and 1.0° of tilt, slower than 0.01 m/s. For 0.5 s. Hand contact is permitted. | The drawer drops. |
| Manufacturing - Task 2<br>Connector Insertion | The plug is seated: within 1 mm along its axis, 0.5 mm off it and 2° of tilt. For 2.0 s. Release is not checked. | The plug falls 0.10 m below its seat; the other plug moves more than 3 mm. |
| Manufacturing - Task 3<br>Power-Driver Screw Fastening | Both bolts are seated in two consecutive physics states. | None. |
| Manufacturing - Task 4<br>Bolt Hand-Threading | The leg's stud is 9.33 mm or more down the hole (thread catch plus one turn), within 0.6 mm and 2°. For 0.5 s. | The leg drops. |
| Manufacturing - Task 5<br>Moving-Conveyor Kitting | A part is in the empty compartment of the moving kit box (within 5 mm of its floor, overhang under 15%). The hands carry at most min(1 N, 25% of its weight). For 0.5 s. | The kit box reaches the belt end, about 23 s after reset. |
| Hotel - Task 1<br>Stable Stacking / Nesting | All dishes are in the destination stack (§12): rim offset under 30 mm, lean under 30°, each dish within 10 mm of its slot height. Peak speed below 0.10 m/s over a 0.25 s window. No robot body over the stack. | A dish drops. |
| Hotel - Task 2<br>Liquid Pour & Serve | 90 mL or more in the glass and less than 5 mL spilled. For 5 s. | 5 mL spilled; the jug drops. |
| Hotel - Task 3<br>Bimanual Singulation | The target utensil is in the bin, no robot body is within 3.5 cm of it, and the other two utensils stay near the support wrist, above the table. For 0.5 s. | A utensil drops below the table. |

**Logistics - Task 1 challenge track.** Logistics - Task 1 also declares a harder rule on the same
episodes: all three bags placed and no box disturbed. An episode runs until all
bags are placed, the first failure or the cap. A failure after the second bag
does not undo the headline success. Each episode records `challenge_success`
next to `success`. The scorer reports the challenge track apart and never puts
it in the headline score. The `l1_headline_bags: 2` setting decides the headline
rule. The runner sets it, and the scorer refuses a row run without it.

**Logistics - Task 4.** The task also records `parcels_delivered`. The scorer does not score it.

## 10. Score

Score a task when every model has run every cell:

```bash
/isaac-sim/python.sh scripts/evaluation/score_benchmark_cells.py score --task h1 \
    --results train=outputs/db_h1_eval_train_gr00t ga_o=outputs/db_h1_eval_ga_o_gr00t \
              ga_s=outputs/db_h1_eval_ga_s_gr00t ga_os=outputs/db_h1_eval_ga_os_gr00t \
    --json outputs/db_h1_score_v1.json
```

Give one `<cell>=<run directory>` per run directory. A cell can repeat: one
directory per model, or per shard. Add `--version v1-plus` to score all rows.
The scorer reads the model of each directory from its `run_manifest.json`.

**Result.** For each model and cell, the scorer reports:

- k (successes) and n (rows of the cell at the version), pooled over the objects
  of the cell;
- the success rate k/n;
- the 95% equal-tailed interval of the posterior Beta(k+1, n−k+1) under a uniform
  Beta(1, 1) prior (the 2.5% and 97.5% quantiles);
- k and n of each object (`objects`);
- the checkpoint, deployment, code revision and evaluation signature of the runs.

For each model with every cell, it reports the macro-average: the unweighted mean
of the cell success rates. `--json` writes the report and the interval convention to a file.
Ablations and trials go under `ablations` and `trials`, never in the headline.

**Status.** The report has one status:

- `complete`: every headline model ran every cell under the approved settings.
- `partial`: a headline model or a cell is missing.
- `non-headline`: a model ran off the approved settings of a cell. The scorer
  scores that cell and lists the deviations. The number is never a headline number.

A deviation of a headline model takes precedence: the report is then
`non-headline`, also when a model is missing.

**A report without every headline model.** The serving code of `pi05` is not
public. Thus a report that you make has no `pi05`
results, and its status is `partial`, or `non-headline` if a run deviates. The
status describes the report, not your model. Read the entry of each model under
`models` in the `--json` report:

- `complete: true` and a `macro_sr`: the model ran every cell of the task.
- `headline: true`: the model ran every cell, and no cell has a deviation.
- `missing_cells`: the cells that the model did not run. A model that you did not
  run lists every cell.

A missing model does not change the numbers of the other models: the scorer
counts k, n and the interval of each model from its own rows. It still holds the
headline models that ran to the same rows and settings, and refuses them if they
differ.

**Refusals.** The scorer refuses the input and exits with code 2 (`refused: ...`) when:

- a directory has no `run_manifest.json` (the baseline run), or names another
  task, cell or frozen set, or a model that the manifest does not list;
- a model has no `--checkpoint` label, or two labels;
- a row of the version is missing, ran twice, or ran another object or other
  draws than its frozen row;
- an episode records no settings or unknown settings, or no success value;
- two headline models ran different rows or different evaluation settings.

The approved settings are the cell's `declared` settings in the manifest: every
environment field, the cap, the physics backend, the cameras, the model's
horizon and the policy rate. The evaluation signature is a hash of what decides
a success rate apart from the scene: the success definition, the cap, the
physics and solver settings, the control timing, and the termination and reset
terms. Do not change environment fields for a scored run.

**Precision.** Half-width of the 95% interval at p = 0.5, the worst case:

| n | 60 (v1 cell) | 120 (v1-plus cell) | 240 | 480 |
| --- | --- | --- | --- | --- |
| half-width | ±12.3 pp | ±8.8 pp | ±6.3 pp | ±4.5 pp |

## 11. Sensitivity sweep

The sweep is a separate robustness section. It never enters the score. It shows
which nuisance factor breaks a policy, not how good the policy is.

- Each task has one sweep of 200 rows: 20 builds of 10 rows. The rows were
  recorded once under the hold-pose policy, and every model replays them.
- Each factor is drawn uniformly, from these ranges:
  - light intensity: ×0.25 to ×1.5 of the scene's lights at the task;
  - colour temperature: 2700 to 6500 K (Hotel - Task 1, Hotel - Task 2: 3000 to 9000 K);
  - head cameras: translation ±30 mm per axis;
  - palm cameras: translation x, y ±5 mm and z (the optical axis) 0 to +5 mm.
- Each camera is drawn independently. The camera translation is in the camera's
  optical frame and adds to its mount. Thus a palm camera stays on its wrist.
- Hotel - Task 3 sweeps the camera translations only. Its scene light has no variation.
- The sweep keeps only rows whose scene came to rest. A visual factor cannot
  change whether a scene comes to rest. Thus the kept values stay uniform.
- Not swept: camera intrinsics, camera rotation, texture and distractors. The
  header of each sweep file and the manifest state the reason for each task.

The light factors are drawn once per build. Thus they rest on 20 independent
draws, each repeated on 10 rows. The camera factors are drawn per row. Read the
light results with that effective sample size.

Run the sweep with `--sweep` in place of `--cell`:

```bash
/isaac-sim/python.sh scripts/evaluation/run_benchmark_cell.py --task h1 --sweep --model gr00t \
    --output outputs/db_h1_sensitivity_gr00t --checkpoint <label of the loaded checkpoint> -- --viz none
```

A sweep always runs all 200 rows. `--version`, `--shard` and `--slices` do not
apply. `score_benchmark_cells.py` does not read sweep runs. Analyse a sweep with
IsaacLab-Arena's sensitivity report. Join the Run's `episode_results_rebuild*.jsonl`
files in the run directory's `arena/` into one file. Then:

```bash
/isaac-sim/python.sh -m isaaclab_arena.analysis.sensitivity.generate_report \
    --episode_results <joined file> --output outputs/db_h1_sensitivity_gr00t.png
```

By default the report analyses every recorded variation, including the training
range's spatial draws. Use `--factors` to select the swept factors.

## 12. Hotel - Task 1 specifics

- **Rows.** The 480 rows are a subset of 1,200 frozen scenes, each drawn from its
  own seed. For each cell and dish, the rows are the 40 scenes with the lowest
  scene numbers. No row matches a draw of the training sets within 1 mm and 0.1°.
- **Success follows the draw.** The scorer judges the stack around the
  destination stack where its draw put it (`success_follows_dest_draw: true`).
- **Rim alignment.** Every cell and the sweep judge alignment at the dish rim
  (`stack_alignment_point: rim`). Results judged at the dish root are a
  different series. Do not pool the two.
- **Render.** `hide_replay_prims: true` hides the prims that the training demos
  were rendered without.

## 13. Why the protocol is so

- **Frozen rows.** Every model replays the same start states. Thus two models'
  results on a cell are a paired comparison, not two samples.
- **60 or 120 rows per cell.** v1 halves the GPU time of v1-plus. v1-plus gives
  the narrower interval (§10, Precision).
- **Beta interval.** The Beta(k+1, n−k+1) interval stays in [0, 1]. Near 0 and 1
  it is asymmetric, where a normal interval is wrong.
- **Unweighted macro-average.** No cell weighs more than another cell.
- **Caps from the training demos.** Every training demo fits in its cap with 25%
  margin or more. Thus a timeout measures a failure, not a slow success.
- **One execution horizon.** No model gains from a longer open loop.
- **Policy at 30 Hz, joint targets at 60 Hz.** The training demos were
  recorded with a joint target at each 60 Hz physics step. A 30 Hz
  control step holds each target for two physics steps. In open-loop replays of
  training demos, the 30 Hz step dropped grasps that the interpolated 60 Hz step kept.
- **Training render.** Every model sees the 640×360 images that its training data
  was rendered at. Another resolution would be a nuisance factor.
- **Sweep apart from the score.** A perturbation answers "which factor breaks
  it". In the score, it would make the result depend on arbitrary ranges.
