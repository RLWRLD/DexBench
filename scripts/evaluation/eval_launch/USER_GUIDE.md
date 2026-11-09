# Eval launcher: user guide

One config runs a model's whole benchmark evaluation on as many GPUs as you give it. The launcher plans the work,
runs it and scores what was accepted. A launch can be stopped at any point and resumed without rerunning finished
work. Three backends run it (`backend.kind`):

- `local` (the one to use on one machine): one driver process on this host places **stations**, a policy server and
  its simulators, on its GPUs in any layout that fits them, and records what counts. Standard library only, no Ray.
- `ray` (several machines, or a Kubernetes cluster as a Ray cluster of pods): the same driver, with Ray placing the
  stations.
- `k8s` (a Kubernetes cluster, as an Indexed Job): one worker per GPU pulling units from a queue directory on a shared
  filesystem.

The launcher is `scripts/evaluation/eval_launch/`; run it with `launch.sh` there, or `python -m eval_launch` with
`scripts/evaluation` on `PYTHONPATH`. It needs Isaac Sim's Python (the simulator image) for runs, and any Python with
PyYAML and this repo's `dexbench_arena` package for planning and rendering Kubernetes manifests.

> **Hardware: use NVIDIA L40S GPUs (46 GB).** This pipeline was verified on L40S only: every launch, rehearsal and
> measured cost so far ran on L40S nodes (AWS g6e: 1, 4 or 8 GPUs per node, one GPU per worker), the local backend
> included (a 4-GPU pod). The unit costs in `timings/` and the per-GPU memory budget (one simulator and its policy
> server sharing a GPU) are L40S figures. Other GPUs have not been tested: a card with less memory can starve the
> renderer (black frames), and a different card changes the costs the planner uses (measure your own: "Plan costs").

## How it works

```
config.yaml ──plan──▶ plan.json ──driver (local, ray) or workers (k8s)──▶ units/<id>/<attempt>/ ──score──▶ score_<task>.json
```

- **Plan** (Kit-free). Reads `benchmark_manifest.json` and the config, then cuts every cell into **units**. A unit is
  one `run_benchmark_cell.py --slices` call over row slices of one cell at the config's `version`, sized to about
  15 minutes. Units run longest
  first. The planner names no task, so a task is planned as soon as its spec is in the manifest.
- **Record (local, ray).** One driver decides which attempt of a unit counts: it records each attempt's start and
  end, and each unit's one accepted attempt, in files created once (`accepted/`, `state/`; "Where things are").
- **Queue (k8s).** The queue lives in `<output>/queue/{todo,claimed,done,failed}`, one JSON file per unit, and holds the
  whole state of a launch. Each claim is a new attempt with its own id and its own output directory. The worker
  heartbeats while the unit runs, and only the attempt that still owns its claim can record a result.
- **Servers and simulators.** Each station (local, ray) or worker (k8s) starts the model's policy server from the
  config's recipe and keeps it for later units of the same checkpoint. Each unit runs in its own Kit process that sees
  one GPU, and is accepted only if all of these hold:
  - the run exited 0 within its deadline;
  - Arena marks every Run of the unit completed;
  - each Run has exactly the episodes its slice names, each replaying a different frozen row;
  - each of those episodes has a video from every camera, and every camera rendered.
- **Score.** Runs the benchmark scorer on the accepted units only.

## Quick start

### On one host

```bash
# 1. Plan and see the estimate. Writes nothing.
scripts/evaluation/eval_launch/launch.sh my_eval.yaml --plan-only
# 2. Run: the driver places a station per layout's worth of GPUs and runs every unit until it is done or failed.
scripts/evaluation/eval_launch/launch.sh my_eval.yaml --local
# 3. After a stop (Ctrl-C, a reboot): accepted units never run again.
scripts/evaluation/eval_launch/launch.sh my_eval.yaml --resume
# 4. Follow, then score.
scripts/evaluation/eval_launch/launch.sh my_eval.yaml --status
scripts/evaluation/eval_launch/launch.sh my_eval.yaml --score
```

`launch.sh` saves your shell's environment for the policy servers before `/isaac-sim/python.sh` adds Isaac Sim's
paths. A model venv started under those paths crashes at start. Start from `examples/single_task_local.yaml`.

With `backend.kind: local` the driver (`python -m eval_launch.local_backend`) runs on this host and needs nothing but
the image's Python. Its GPUs are `backend.gpus`: a list of device ids, a count (the first that many of those
`CUDA_VISIBLE_DEVICES` or `nvidia-smi` shows), or, left out, every GPU this host shows. Station `i` takes the `i`-th
run of the layout's GPU count of them. Each station's server is a child process of the driver; each unit runs in a
child process of its own on its slot's GPU, which the driver kills if it does not answer long past its deadline. The
children die with the driver, and what a killed one left (a Kit, a server) carries the launch's owner tag and is
stopped by the next driver. A local launch started before the driver existed (its units in `queue/todo`) resumes the
way it started, one worker per GPU.

### Stations and layouts

`servers.layout` names where a station's server and simulators go, for `local` and `ray` (`k8s` runs `colocated`
only):

| `servers.layout` | GPUs per station | Where the processes go |
|---|---|---|
| `colocated` (default) | 1 | server and one simulator share the GPU, as the other backends run them |
| `server+1` | 2 | server alone on one GPU, one simulator on the other (Cosmos3: its server takes 33 GB of an L40S) |
| `server+N` | 2 | server alone, N simulators sharing the second GPU |
| `{sim_gpus: G, sims_per_gpu: N}` | 1 + G | server alone, N simulators on each of G GPUs |
| `{sim_gpus: 0, sims_per_gpu: N}` | 1 | server and N simulators share the GPU (`colocated+N`; state `gpu_memory_gb.server` too) |

`servers.max_sims_per_gpu` (3 by default) caps N: a layout with more simulators on one GPU is refused. On an L40S,
4 simulators peaked at 33 GB with the GPU saturated (~93% utilisation) and steps 1.5-2.6x slower, against 1.3-2.1x at 3
(L40S measurement, 2026-11-05); another card may state its own cap. A unit's deadline is multiplied by N, the
simulators sharing its GPU.

Each simulator is its own Kit process (`run_benchmark_cell.py`, started exactly as the k8s workers start it) and sees
only its own GPU (`CUDA_VISIBLE_DEVICES`: the local driver sets it per process, Ray per actor). Nothing holds a process
to its share of a GPU's memory, so a layout with several simulators on one GPU must state `servers.gpu_memory_gb`
(`server`, `sim`) and `backend.gpu_memory_gb` (the card), and is refused when they do not fit; a Cosmos3 server with a
simulator beside it on one L40S is refused that way (`{server: 33, sim: 14}` on 46 GB).

### The tree a launch runs from

Every simulator runs `run_benchmark_cell.py` from one checkout of this repo (one tree per launch), with its
submodules, its assets and the benchmark's runtime inputs. On one host that is your checkout; on a cluster it is a
copy on the shared volume, staged once per commit. Build it from the published sources with one command, on a host
that mounts the volume, as your normal (non-root) user, not from inside a container or pod: files a container writes
into `assets/` belong to its user (often root), and the next `git lfs pull` there then fails.

```bash
scripts/evaluation/eval_launch/stage_tree.sh <commit> <volume>/<you>/trees/<12-char commit>
```

It clones the repository your checkout came from (its `origin` remote; `--repo <url>` names another), so the commit must
be one that repository has. It needs git, git-lfs, the `hf` CLI and python3, and the credentials git and `hf` already
use (a git credential helper for GitHub and huggingface.co, `HF_TOKEN` or `HF_TOKEN_PATH`, `hf auth login`; an SSH
`origin`, such as `git@github.com:...`, needs an SSH key for it on the staging host, or pass `--repo https://...`), with
a token that can read the assets and datasets repositories; it never prints one. It refuses to run as root, and an
existing destination. It builds in `<dest>.partial` and renames it to `<dest>` only when every step passed, which are
the manual steps:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout <this repo> <tree> && cd <tree>
git lfs install --local --skip-smudge --skip-repo     # before the checkout: no pointer is smudged at it
GIT_LFS_SKIP_SMUDGE=1 git checkout --detach <commit>
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive submodule/isaaclab_arena
git lfs pull --include="assets/**" --exclude=""      # the assets' payloads, from the Hugging Face LFS endpoint in .lfsconfig
./scripts/download_datasets.sh --eval-conditions     # outputs/db_conditions/ (the frozen rows, each checked)
                                                      # and outputs/db_h3_inputs/ (Hotel - Task 3's input package)
git rev-parse HEAD > .source-commit                   # the commit the pods hold the tree to
git ls-files ... | xargs sha256sum > SHA256SUMS       # every tracked file but the submodules' gitlinks
```

On the one cluster that built a tree this way (an AWS L40S cluster, at a commit whose `--eval-conditions` did not
yet install Hotel - Task 3's inputs) it took about 17 min and 11 GB (clone 2 min, submodules 6 min, assets 4.5 min, frozen
rows 3.5 min); every Hotel - Task 3 unit of that tree then failed at its start for want of
`outputs/db_h3_inputs/manifest.json`. The launcher now refuses to plan a launch on a tree that lacks a file a planned
task reads (each planned cell's rows, and the runtime inputs a task's spec declares), naming the file and the command
that installs it; `--plan-only` notes them. On Kubernetes, when this host sees `backend.k8s.tree_dir`, `--k8s-job`
checks that tree the same way, and its `.source-commit`, before it writes the Job ("... lacks N files the planned tasks
read ...: run ... in it", or "... is staged at <commit> ..."). `.source-commit` must hold the full commit: the pods
refuse any other tree (`backend.k8s.commit`). A tree that differs from a staged one only under
`scripts/evaluation/eval_launch/` can be a hard-link copy (`cp -al`) with that directory replaced.

### On Kubernetes

What a cluster decides (its image, namespace, shared volume, node rules, pod sizes, required labels, annotations and
security contexts, whether DaemonSets are allowed) goes in a **site profile**, a file a launch config names with
`backend.site`; the config keeps what the launch decides (the model, the tasks, the GPUs, its Job name, its tree).
`examples/sites/kubernetes.yaml` lists every key with a plain cluster's values; `examples/kubernetes.yaml` (k8s
backend) and `examples/ray_kubernetes.yaml` (Ray) are launches that name it. The config's own values win over the
profile's; mappings merge key by key. A profile that sets a launch's own field (`job_name`, `tree_dir`, `commit`) is
refused, and `--k8s-job` writes the merged config, so the pods never read the profile file.

1. **Stage the tree** on the shared volume: `stage_tree.sh <commit> <volume>/<you>/trees/<12-char commit>` ("The tree a
   launch runs from").
2. **Fill in the config.** Copy an example and `examples/sites/kubernetes.yaml` to `configs/local/` (git ignores it: real
   image, namespace, volume and checkpoint names stay on your machine), fill in the profile's `<...>`, point the
   config's `backend.site` at your copy (a relative profile is looked for beside the config first, then in the
   package, so an example's `examples/sites/kubernetes.yaml` would still find the shipped placeholders, which
   `--k8s-job` refuses), set `backend.k8s.tree_dir` and `commit` to that tree, then render the manifests:

   ```bash
   scripts/evaluation/eval_launch/launch.sh my_eval.yaml --k8s-job jobdir/
   ```

   This writes `jobdir/job.json`, `jobdir/prepull.json` and `jobdir/eval.yaml`.
3. **Copy the config** `jobdir/eval.yaml` to `<output>/eval.yaml`. The pods read it from there.
4. **Pre-pull the image** before a launch of more than a few GPUs. A cold pull from another region takes 13–20 min
   per node. Where you may not create DaemonSets, set `prepull: false` in the profile: `--k8s-job` then writes no
   `prepull.json`, and each new node pays the pull inside the launch's own pods (allow for it in `deadline_h`):

   ```bash
   kubectl apply -f jobdir/prepull.json
   kubectl rollout status ds/<job_name>-prepull
   kubectl delete -f jobdir/prepull.json
   ```

5. **Start:** `kubectl apply -f jobdir/job.json`. Pod 0 writes the plan and queue; every pod runs one worker.
6. **Follow:** `launch.sh my_eval.yaml --status`, from any machine that sees the volume.
7. **Score:** `launch.sh my_eval.yaml --score` once `--status` shows nothing left to run.

Check free GPUs first (`kubectl get pods` with GPU limits). On a shared cluster, leave headroom for others.

What the one second cluster tried so far needed of its profile (an AWS L40S namespace with admission rules of its
own): a Ray head with no GPU toleration and its own CPU-pod sizes (`ray.head_tolerations: []`, `head_cpu`/`_limit`,
`head_memory`/`_limit`), provenance `annotations`, a container `security_context` (`allowPrivilegeEscalation: false`;
not `capabilities: {drop: [ALL]}`: the image runs as root and root without capabilities cannot read `/isaac-sim`,
so Kit finds no extensions), a pod `seccompProfile`, and `prepull: false`. A cluster's label rule is
`required_labels` (a config whose labels lack one is refused). A rule the launcher cannot state (e.g. where outputs
must live) goes in the profile's `note`.

### On Ray (several machines, or a Kubernetes cluster)

`backend.kind: ray` runs the same driver with Ray deciding where each station goes: a Ray placement group of one-GPU
bundles on one node, its server and each simulator slot an actor. `launch.sh <config> --local` or `--resume` starts
the driver (`python -m eval_launch.ray_backend`) for a Ray config. On one machine, `local` does the same without Ray.

```bash
scripts/evaluation/eval_launch/launch.sh my_ray_eval.yaml --plan-only
scripts/evaluation/eval_launch/launch.sh my_ray_eval.yaml --local      # the driver: plans, places stations, runs every unit
scripts/evaluation/eval_launch/launch.sh my_ray_eval.yaml --resume     # after a stop: accepted units never run again
scripts/evaluation/eval_launch/launch.sh my_ray_eval.yaml --status     # todo / running / done / failed, and why each failed
scripts/evaluation/eval_launch/launch.sh my_ray_eval.yaml --score      # the accepted attempts only
```

Without `backend.ray.address` the driver starts a Ray instance on its own host with `backend.gpus` (a count, or the
list of GPU ids to use). With an address (`auto`, or `<head>:6379`) it joins a running cluster: start one with
`ray start --head --num-gpus=<n>` on one node and `ray start --address=<head>:6379` on the others (pods, SLURM nodes),
each with the launch's output directory mounted at the same path. Ray needs to be importable by the driver and on
every node (`ray==2.58.1`; the simulator image's recipe, `docker/Dockerfile`, installs it into Isaac Sim's
Python; on an image built before it did, install it with `pip install --no-deps --target <dir> ray==2.58.1` and put
`<dir>` on `PYTHONPATH`, or name `<dir>` in `backend.k8s.ray.pythonpath` on Kubernetes, as Isaac Sim's Python already
has Ray's dependencies). See `examples/ray_local.yaml`.

**Bringing your model onto Ray: layout, memory, smoke check.** A layout is where one server and its simulators go
("Stations and layouts"). Pick it by what the server takes on one GPU:

- `colocated`: the server and one simulator share a GPU. The default, and the safe start: RLDX, GR00T and pi0.5 run
  so on an L40S.
- `colocated+N` (`{sim_gpus: 0, sims_per_gpu: N}`): the server and N simulators share a GPU; fewer GPUs per unit of
  work when the server is small and served requests one at a time (the server kit does; a shared server queues the
  simulators' requests, so N is limited by how fast it answers too).
- `server+N`: the server alone on one GPU, N simulators on a second; for a server too big to share (Cosmos3 takes 33 of
  an L40S's 46 GB).

Nothing holds a process to its share of a GPU, so a layout that shares one states what fits: `servers.gpu_memory_gb`
(`server`: one server's peak, `sim`: one simulator's) and `backend.gpu_memory_gb` (the card's), checked per GPU before
anything starts, and `servers.max_sims_per_gpu` (3 by default) caps N. To measure the figures and see whether two
simulators fit:

1. Run `launch.sh my_eval.yaml --smoke` with `layout: colocated`. It runs one station only, `--smoke-rows` rows (2 by
   default) on each of its simulator slots at once, through the config's own driver (`local`, or `ray` with a Ray
   instance on this host or `backend.ray.address`), into `<output>-smoke/<time>/`, and never touches `<output>`.
   Every unit's outcome now records `gpu_memory`, the peak `nvidia-smi` reported for the simulator's whole GPU while
   it ran (its server and neighbours included); the check prints it per unit and writes all of it to `smoke.json`.
2. Run it once more with `layout: server+1`: the simulator is alone on its GPU, so its peak is `sim`, and the
   colocated peak less `sim` is `server`. State them in `servers.gpu_memory_gb`, choose the shared layout, and run
   `--smoke` with it.
3. Launch when it passes. Each unit's item is `pass`, `fail`, `advisory` or `not measured`, and the check fails only
   on a `fail`: **frames**, the hard gate (a unit not accepted: an episode missing, a camera that never rendered; the
   cause and the unit check's problems are listed); **timing** (a unit's slowdown, its stepping time over its
   simulated time, against the plan's for its task with N simulators on a GPU: up to 1.5 times passes, up to the
   plan's `deadline_factor` (3) is `advisory`, measure your costs ("Plan costs"), and past it fails, as the launch's
   units would be stopped at their deadline; the smoke's slowdown is a mean over a few rows, and without
   `plan.timings` the plan's is the default for GR00T on an L40S; a unit with no step timings fails it, since the
   check could not run); and **memory** (a GPU's peak above 95% of the card,
   where the renderer starves and frames go black, or more than 10% above what `gpu_memory_gb` states for every
   process on that GPU; a GPU `nvidia-smi` could not read is `not measured`). The memory reading is a floor: each
   smoke unit builds one Run of one cell (the one whose longest Run has the most rows), while a launch's units build
   up to 60 Runs in turn in one Kit process, and other tasks' scenes differ. A pass says nothing about those; leave
   the declared figures some room, as the report and `smoke.json` repeat. `--smoke` exits 0 when the check passed, 1
   when it ran and failed, 2 when it could not run (a refused config, a tree without its inputs, a driver under which
   no unit ran) and 130 when a signal stopped it, through `launch.sh` (`/isaac-sim/python.sh` run directly turns
   every non-zero status into 1; `launch.sh` reads the launcher's own). A smoke whose driver ran writes `smoke.json`
   with its `result` (pass, fail, not run, stopped); one refused before that (a config, a tree without its inputs, no
   plannable cell) writes none, so read the exit code, not the newest `smoke.json`.
   `--tasks` names the tasks it picks its one cell from.

`examples/recipe_template.yaml` is a Ray config to start from: a model the config registers, served by the kit, with
every field the worker fills into a recipe explained (`{checkpoint}`, `{checkpoint_id}`, `{port}`, `{token_file}`,
`{worker_dir}`, `{launcher}`).

**A Ray launch on Kubernetes** (no operator: plain pods running `ray start`). Give the Ray config a `backend.k8s` block
as for the k8s backend (from its site profile: namespace, image, node selector, tolerations, volume, labels, and
`cpu`, `memory` and `shm_size` per GPU; from the config: the staged tree and the Job's name), optionally with `backend.k8s.ray` (`ray_k8s.RAY_K8S_KEYS`: GPUs per worker pod, the head
pod's size, restarts, where Ray is when the image lacks it). Then `--k8s-job jobdir/` writes `jobdir/ray.json`:

- a **head** Job: one pod, no GPU, on the nodes the config selects. It starts the Ray head and runs the driver
  (`ray_backend --ray-address auto`), which plans the launch or adopts its plan; it is the only writer of the launch's
  record. Once every unit has ended (the driver exits 0, or 3 when some failed for good) it writes
  `<output>/k8s/<launch id>.finished` and exits; any other driver exit fails the pod and the Job starts a new head.
- a **workers** Job: indexed, `backend.gpus / worker_gpus` pods of `worker_gpus` whole GPUs each (default: one
  station's GPUs; 8 fills a node). Each joins the head as a Ray node advertising its GPUs; Ray gives each server and
  simulator actor its own `CUDA_VISIBLE_DEVICES`. A worker leaves when the head's marker appears, or earlier, once
  its driver has released it (below). The Job replaces a pod only once it has ended (`podReplacementPolicy: Failed`),
  so an index never has two pods.
- a headless **Service** `<job_name>-head` the workers find the head by.

Steps 3 to 7 above apply as they are (copy `eval.yaml`, pre-pull, `kubectl apply -f jobdir/ray.json`, `--status`,
`--score`). The policy servers start in the head pod's environment with their secrets from the worker pod's own:
`backend.k8s.secret_env` goes to the worker pods (`secretKeyRef`), and Ray carries the variables' names only, never a
value. What happens when a pod dies:

- **a worker pod**: its actors die; each attempt they held ends `lost` (charged), the pod comes back under its index
  (`worker_restarts`), Ray re-places its stations' bundles there, and what the dead actors' Kits left running is stopped
  by the next actor on the pod.
- **a released worker pod**: not a failure. Once no unit is left to start, the driver tears down the stations with
  nothing to do (it keeps as many idle ones as the running units would fill, for their retries; see "What happens
  when something fails") and then lets go of each worker pod that hosts none of the kept stations: it places a Ray
  placement group of the pod's own resource and all its GPUs on the pod's node (which Ray places only once nothing
  else holds a GPU there, and which keeps anything else off it). It checks that the fence stands on the pod's current
  Ray node, and only then writes `<output>/k8s/<launch id>.release/<pod name>`, naming that node (the tag each join
  of the pod's node advertises). The pod's watcher sees the file within `release_poll_s` (5 s) and, if it names its
  current node, stops its Ray node and exits 0, so its index succeeds and is not restarted; its GPUs go back to the
  cluster while the last units run elsewhere. Every pass checks the fences again: one that no longer stands while
  its pod is up is dropped and its release taken back, and the pod is fenced and released again later. A new driver
  (a new head) fences the pods an earlier one released before it places any station, and refuses to start (the Job
  then starts another head) when one is still up and cannot be fenced within 300 s.
- **the head pod**: the Ray cluster goes with it (the workers' Ray nodes stop and keep trying to join again). The Job
  starts a new head (`head_restarts`); its driver waits for the dead driver's lock on the shared volume to go (exit 75,
  retried), ends what the dead one left (`recover`: accepted from an outcome its actor wrote, else `uncertain`, charged)
  and runs the rest on the rejoined workers.

To stop a Ray launch on k8s, delete the head Job first and the workers Job after it: the driver hands its running
units back untried on SIGTERM, while a worker deleted first takes its units down as `lost`, charged (test,
2026-11-05). `examples/ray_kubernetes.yaml` is a Ray launch on a cluster's site profile.

What the record holds instead of a queue: `accepted/<unit>.json` (the unit's one accepted attempt, created once, with
every charged attempt's outcome; `--score` reads it) and `state/` (each attempt's start and end, one file each, created
once; `state/driver.lock`, held by the running driver). `queue/done/` and `queue/failed/` are written as views of them
for readers of the queue's layout. The `local` and `ray` drivers keep the same record, so a launch may resume under
either; `k8s` refuses a driver's output and the drivers refuse a queue's.

## Plan costs

The planner sizes units and estimates a launch from per-task costs. `plan.timings` names a file of measured ones
(relative to the launcher's directory; `--timings <file>` overrides it): `tasks.<task>.slowdown`, the median control
step's wall time over its simulated time (1/60 s), and `tasks.<task>.build_s`, a unit's wall time not spent stepping
or starting the server, per Run. `timings/l40s_example.json` is one such file, measured on L40S GPUs with one
simulator and a colocated server per GPU: it is right only for that card and layout. A task without a measurement
takes the median of the measured ones; without a file every cost is a default (`plan.DEFAULT_COSTS`), and
`--plan-only` says which source each task's cost came from.

On another GPU, run a few units first and measure your own: every unit's outcome records `wall_s`, `stepping_s` and
`steps` (Arena's step timings) and `server_load_s` (in `accepted/<unit>.json` on `local` and `ray`, `queue/done/` on
`k8s`). A layout that puts N simulators on one GPU runs each up to N times slower; the deadline allows for it
(`deadline_factor`), the estimate does not.

## Resuming a launch

Resuming is built in. A launch's files are its whole state: with `local` or `ray`, an accepted unit never runs again
and the next driver ends what the stopped one left (accepted when its slot had written a checked, complete outcome,
charged as `uncertain` otherwise); with `k8s`, a unit in `done/` never runs again, and a unit that was running when
the launch stopped goes back to `todo/`.

| What stopped it | What to run |
|---|---|
| `local` or `ray`: Ctrl-C, a reboot, a lost host, a killed driver | `launch.sh my_eval.yaml --resume` (`--retry-failed` for another round of failed units) |
| A deleted, failed or deadline-ended Job | `kubectl delete job <name>` if it still exists, then `launch.sh my_eval.yaml --resume --requeue-claims`, then `kubectl apply -f job.json` |
| Some units in `failed/` you want to give another round (e.g. after fixing the recipe) | add `--retry-failed` |
| A single lost pod in a running Job | nothing: the index restarts (`retries_per_worker`), and its claim is requeued once its heartbeat is 10 min stale |

The rest of this section is the `k8s` queue's.

- **`--requeue-claims`.** Requeues every claimed unit at once. Use it only when no worker is running: after
  `kubectl delete job`, wait until its pods are gone (`kubectl get pods -l job-name=<name>`), since terminating pods
  still run their worker (2026-11-02: two such pods left a duplicate claim). Without it, a claim goes back to the
  queue once its heartbeat is 10 min stale, which is safe while workers still run.
- **What a resume may change.** It may change anything that doesn't change the planned units: a fixed server
  recipe, a newer tree, another GPU count or backend. `--resume` writes those settings into `plan.json`, which is
  where workers read them when they start, so workers started after the resume use them. A change to `model`,
  `tasks`, `cells`, `sweep`, `version` or `plan` is refused; start a new output directory for that. The version is
  compared by what it means: an omitted `version` is `v1`, and a plan made before versions (no `eval_version` in
  `plan.json`) ran `v1-plus`, so its config resumes or extends it with `version: v1-plus`.
- **What a resume repairs.** Every planned unit missing from the queue (a start that died half way) is queued again.
  A queue step that died half way is cleaned up once stale: removed if it had already recorded its result, requeued
  if it hadn't. So no unit is lost, and no accepted unit runs again.
- **What stays.** Every attempt's output, including failed ones, stays under `units/<id>/`. The done record names the
  accepted attempt, and only that one is scored.

## Config reference

```yaml
model:
  key: rldx11                    # a key of benchmark/models.py: the client block the simulator uses
  checkpoint: /path              # one checkpoint for every task, or:
  checkpoints: {l1: /path, l2: /path}   # one per task
  # A model of your own, registered by this config (benchmark/models.py config_model): a key models.py lacks and
  label: My Model                # its label,
  group: community               # the only group a config model takes (the default): reported apart, never the headline,
  policy: {type: ..., ...}       # and its client's policy block (CONNECT_YOUR_POLICY.md, step 2)
tasks: [l1, l2]                  # or: all (every task the manifest can run; the others are listed with a reason)
cells: all                       # or a list for every task, [train, ga_s], or per task, {l1: [train]}
sweep: false                     # true is refused: the launcher runs cells only (use run_benchmark_cell.py --sweep)
version: v1                      # v1 (default): 60 fixed rows of each cell; v1-plus: all 120 (docs/Eval_Protocol.md §1)
servers:
  layout: colocated              # local and ray: colocated | server+N | {sim_gpus: G, sims_per_gpu: N}
  max_sims_per_gpu: 3            # local and ray: the most simulators a layout may put on one GPU (L40S: 3)
  gpu_memory_gb: {server: 33, sim: 14}   # local and ray: what one server and one simulator take, checked per GPU
  recipe:                        # fields: {checkpoint} {checkpoint_id} {port} {token_file} {worker_dir} {launcher},
                                 # shell-quoted already; str.format, so a literal brace is doubled: ${{HOME}}
    setup: ...                   # worker's environment, before the server, bounded; exit non-zero = refuse
    command: ...                 # the server, in the saved server environment, on the worker's GPU
    ready: Server ready          # regex on this start's server log
    probe: ...                   # optional: a request that succeeds only when the server serves
    client_set: [key=value]      # optional: --set for the simulator's policy block (e.g. a per-worker contract path)
plan:
  timings: timings/l40s_example.json   # measured costs, relative to the launcher; without it, defaults
  unit_minutes: 15               # target unit length
  deadline_factor: 3             # a unit is stopped at max(min_deadline_s, factor x its planned worst case);
                                 # local and ray: times the simulators sharing its GPU (servers.layout)
  min_deadline_s: 1800
  max_attempts: 2                # a failed or lost unit runs this many times in all (driver launches: per row,
                                 # the episodes a crashed attempt finished are kept; see "Salvaged episodes" below)
  server_ready_timeout_s: 1800   # setup plus server start
backend:
  kind: local | ray | k8s
  gpus: 4                        # or a list of GPU ids for local and a local Ray instance: [0, 2, 3]; local: may be
                                 # left out (every GPU CUDA_VISIBLE_DEVICES or nvidia-smi shows)
  gpu_memory_gb: 46              # local and ray: one GPU's memory (46 on an L40S), for the layout check
  local:                         # local backend
    port_base: 5810              # station i's server listens on port_base + 10 i
    slot_restarts: 3             # a slot that loses its process this many times in a row gets its station rebuilt
    station_restarts: 2          # a station that cannot run units is rebuilt this many times, then retired
    setup_hook: pkg.module.fn    # optional: called first in every server and slot process (tests use it)
  ray:                           # Ray backend
    address: auto                # a running cluster; none: start Ray on this host with backend.gpus
    num_cpus: 32                 # that instance's CPUs (Ray's count; the simulators are not limited by it)
    object_store_gb: 1           # that instance's object store (it carries unit specs and outcomes only)
    runtime_env: {env_vars: {}}  # Ray's runtime_env for its workers
    port_base: 5810              # station i's server listens on port_base + 10 i
    actor_restarts: 3            # Ray restarts a dead server or simulator actor this many times
    station_restarts: 2          # a station that cannot run units is rebuilt this many times, then retired
    secret_env: [HF_TOKEN]       # the servers' secrets, by name: each Ray node's environment holds the values
    release_dir: <dir>           # where the driver names the worker pods it lets go (set by the k8s head with
                                 # --release-dir; none elsewhere, and then no machine is let go)
  pythonpath: [/path/client]     # model client packages the simulator imports (e.g. RLDX's)
  env: {HDF5_USE_FILE_LOCKING: "FALSE"}   # added to each simulator's environment (here: an NFS volume)
  site: <cluster>.yaml           # optional site profile (eval_launch/site.py): the cluster's image, namespace, storage,
                                 # node rules, pod sizes, annotations, security contexts; this config's own values win
  image: <registry>@sha256:...   # k8s
  k8s:
    namespace: ...
    job_name: ...
    tree_dir: /data/.../trees/<commit12>
    commit: <full commit>
    volume_claim: ...
    volume_mount: /data
    cpu: 10                      # required: each worker pod's share of its node (Ray backend: per GPU of a worker pod)
    memory: 80Gi                 # required (Ray: per GPU)
    shm_size: 16Gi               # required: the pod's /dev/shm (Ray: per GPU)
    env: {MODEL_OUTPUT_DIR: ...} # the cluster's own pod variables
    secret_env:                  # pod variables from Kubernetes secrets, never written into a file
      HF_TOKEN: {secret: hf-token, key: token}   # gated Hub models; create the secret once (below)
    labels: {cost-center: evals}   # on the Job, its pods and the pre-pull
    required_labels: {cost-center: evals}   # the cluster's label rule: refused unless labels has them
    annotations: {...}           # on every Job and pod template (e.g. a cluster's provenance keys)
    security_context: {allowPrivilegeEscalation: false}   # the container's, if admission asks (not drop: [ALL]: Kit fails)
    pod_security_context: {seccompProfile: {type: RuntimeDefault}}                     # the pod's
    python: /isaac-sim/python.sh # the image's Isaac Sim interpreter
    prepull: true                # false: --k8s-job writes no prepull.json (a namespace that may not create DaemonSets)
    retries_per_worker: 2        # pod restarts per index
    deadline_h: 24               # the whole Job
    ray:                         # Ray backend on k8s, all optional (ray_k8s.RAY_K8S_KEYS)
      worker_gpus: 8             # whole GPUs per worker pod (default: one station's)
      head_cpu: 2                # the head pod (no GPU): Ray's head and the driver
      head_memory: 8Gi
      head_cpu_limit: 4          # limits, where the cluster wants them stated (default: the requests); never below
      head_memory_limit: 16Gi    # the requests, which Kubernetes refuses
      head_tolerations: []       # the head's own (default: tolerations below); [] where a GPU toleration needs a GPU
      head_restarts: 3           # a dead head is replaced; its driver recovers the launch
      worker_restarts: 3         # per worker index
      release_poll_s: 5          # how often a worker looks for its release file (0.1 to 60)
      pythonpath: [<dir>]        # where Ray is, for an image without it (pip install --no-deps --target <dir>)
    node_selector: {...}
    tolerations: [...]
output: /data/.../launches/<name>  # on a filesystem every worker sees
```

**Hub token on Kubernetes.** Some models load files from gated Hub repos (GR00T's and Cosmos3's backbones). Give the
pods a token through a Kubernetes secret, created once per namespace by whoever owns the token, then name it in
`backend.k8s.secret_env`:

```bash
kubectl -n <namespace> create secret generic hf-token --from-file=token=<file holding the token>
```

The Job then sets `HF_TOKEN` from the secret: the token is in no config, Job file or log. The policy servers get it
with the rest of the pod's environment, which the pod saves to its own `/tmp/server.env` (owner-only, in the
container, not on the shared volume) before Isaac Sim's launcher changes it; the worker removes that file once it
has read it. On a host, `launch.sh` saves the shell's environment the same way and the launcher removes the file as
it starts; each local server process (or worker) gets a copy it removes as it reads it.

On Ray the servers' environment reaches each server as an argument of its actor, which Ray keeps on the head, so the
secrets go by name only: the variables of `backend.k8s.secret_env` and `backend.ray.secret_env`. The server actor
reads their values from its own process, which inherits them from where its Ray node started: a k8s worker pod (the
secret), the driver for a Ray instance it starts on its host, and for a cluster started by hand the shell each
`ray start` ran in, so set them there. A node without one refuses to start the station's server and says which.
Without a token, keep `HF_HUB_OFFLINE=1` in the recipes and the models in the shared Hub cache.

**Server recipes.** Put every check that can fail before a unit runs into `setup`: the checkpoint's hash, the
server runtime's commit, and that the simulator can import the model's client. If setup fails, the unit goes back
untried and the worker stops, so no unit is spent on a broken environment. Anything a recipe writes for the
simulator goes under `{worker_dir}`, keyed by `{checkpoint_id}` when it depends on the checkpoint; several workers on
one host must never share a file. Each step is filled in with `str.format` before `bash -c` runs it: write a brace the
shell should see doubled (`${{HOME}}`), and do not quote a field again (`{checkpoint}`, not `"{checkpoint}"`), since
its value is shell-quoted already. `--start`, `--local`, `--resume` and `--k8s-job` refuse a config whose recipe has
no `command` or does not format, before anything is queued. The ready line is looked for in what the server prints,
after setup's output. The examples name no cluster, volume, checkpoint or run:
keep a copy with your real values in `outputs/configs/`, which git ignores and the simulator image mounts (the
image's own copy of the source does not see a config written after it was built). A model's server comes from the model's own repository: the launcher carries
none (the server kit, `source/dexbench_policy_server`, is for a model of your own, CONNECT_YOUR_POLICY.md).

**Instructions.** The client sends the environment's own task text (Arena's `get_language_instruction()`), never the
checkpoint's, and the launcher does not compare it with the training set's. Every unit's outcome lists the distinct
instructions its episodes recorded. A config that still has the retired `instructions:` key is refused on a new launch;
a launch planned with it resumes (`--start`, `--local`, `--resume` on its output directory) and the key is ignored.
`--k8s-job` cannot see the volume, so it refuses the key too unless `--existing-plan` says the output already has a
`plan.json`; it then leaves the key out of `eval.yaml`.

## What happens when something fails

| Failure | What the launcher does |
|---|---|
| `k8s`: the server won't start, setup fails or runs past its time, or the probe fails | The unit goes back untried and the worker stops; the pod's index retries (a local launch from before the driver: the launcher restarts the worker). |
| `k8s`: the worker's own environment raises (filesystem, imports) | Same as above: the unit is not charged an attempt, and the traceback is in the worker's log. |
| The unit's run exits non-zero, a Run did not complete, an episode is missing or replayed twice, or a camera never rendered | The unit fails this attempt and runs once more (`max_attempts`). The outcome keeps the cause: the exit code, the unit check's problems and the last 20 lines of the attempt's log. It stays in `failed/`, with every outcome, after the last attempt. A unit accepted on a later attempt is scored from that attempt, and `--score` reports it (below). |
| The unit's metadata (timings, composed config) is unreadable | Recorded in the outcome; it does not stop the worker. |
| A unit hangs | It is stopped at its deadline (its whole process group) and counts as a failed attempt. |
| A worker or pod dies mid-unit | Its claim is requeued once the heartbeat is 10 min stale (measured on the shared filesystem's clock, so a pod's own clock does not matter), and that counts as an attempt. One worker (the sentinel) stays until every unit has ended. An idle worker leaves only after it has seen the sentinel renew its lease; a sentinel that stops renewing is replaced. |
| `local` or `ray`: no unit is left to start while some still run | The driver releases the stations with nothing to do so their GPUs go back, instead of holding every station until the last unit ends. It keeps as many idle stations as the running units would fill (one per unit and simulator slot, rounded up), so a unit that fails, or whose station fails, starts again at once; nothing goes unless all of those are there and answer. A station goes only when it holds no attempt and its processes, asked at that moment, answer that they run nothing; it is marked retiring first (no unit goes to it), checked again, then torn down. A pass runs at most every 30 s, in a launch's tail only. `local` frees the station's GPUs and lets go of no machine; `ray` on Kubernetes then lets go of each worker pod that hosts no kept station (above). A launch whose stations are all retired or released with units left stops with exit 1, for `--resume`. A lost machine that held the running unit and its spare waits for its pod to come back, as without releases. |
| A slow or restarted worker finishes after its claim was requeued | Its result is recorded as superseded and never scored, even under the same worker name. |
| `local`: a unit's slot process dies mid-unit | The attempt is `lost` and charged; what it started (its Kit) is stopped by its owner tag before the slot takes another unit. |
| `local`: no result long after the deadline (a frozen slot) | The driver revokes the attempt 30 min past its deadline (on its own monotonic clock), SIGKILLs the slot process and stops its Kit; a revoked attempt is never accepted. |
| `local` or `ray`: the server will not start, or the cell runner refuses the unit | The unit goes back untried (`released`, not charged); the station is rebuilt (`station_restarts`), then retired. With every station retired the driver stops; `--resume` once fixed. |
| `local`: the driver is killed | Its server and slot processes die with it (`PR_SET_PDEATHSIG`); what they started is stopped by the next driver's sweep. The next driver (`--resume`) accepts an attempt whose slot had written a checked, complete outcome, charges one it cannot tell (`uncertain`), and runs the rest. A second driver on the same launch is refused while one runs. |
| Ray backend: a simulator's actor, its Kit or its node dies mid-unit | Ray reports the actor dead (a process at once, a node once Ray's own health check gives it up; no file heartbeat, no clock compared); the attempt is `lost` and charged, Ray restarts the actor in its place, and the slot takes a unit once the actor answers again. What the dead actor started is stopped by the next actor on the node and by the driver's sweep at start and stop. |
| Ray backend: no result long after the deadline (a frozen actor) | The driver revokes the attempt 30 min past its deadline (on its own monotonic clock), kills the actor (Ray restarts it), and a result that still arrives is fenced out: an attempt ends once, and a revoked one is never accepted. |
| `local` or `ray`: a unit's simulator crashes, hangs or is lost after finishing some episodes | Episode-level salvage: each episode it finished (its Arena result line, and a complete video from every camera) is accepted for its row; the row under way is charged, not the unit, and the unit's next attempt runs only the rows left (`--slices`). A row charged `max_attempts` times fails alone; the unit then ends `failed` with its failed rows listed (`--status`, `queue/failed/`), and `--retry-failed` runs those rows again. When the attempt never got into an episode (Kit or a Run's environment failed to start), every row it was given is charged, as before. |
| Ray backend: the driver is killed | Ray stops the launch's actors with it. The next driver (`--resume`) accepts an attempt whose actor had written a checked, complete outcome, charges one it cannot tell (`uncertain`), and runs the rest. A second driver on the same launch is refused while one runs. |
| `k8s` (and a local launch from before the driver): a worker gets SIGTERM (Ctrl-C or `kill`, a pod deleted or evicted) | It hands its unit back untried first, then stops its simulator and server; k8s pods get 120 s for this (`termination_grace_s`), and the local backend kills what is left after 2 min. The worker and the local launcher run under `signal_group.sh`, because `/isaac-sim/python.sh` passes no signal to the python it starts. Before that, a deleted pod's worker never saw the SIGTERM and a `kill` of the local launcher left its workers running (rehearsal, 2026-11-02). A forced (SIGKILL) delete leaves the claim until the 10 min stale requeue. On a host, a worker killed that way also leaves its simulator and server running; the worker started in its place stops them first (they carry the worker's tag, `EVAL_LAUNCH_OWNER`). |

## Where things are

```
<output>/
  eval.yaml                      # the config the pods read (k8s)
  launches/<UTC time>-<id>/      # one per start (--start, --local, --resume, a driver's start; --k8s-job writes it
                                 # beside its manifests, copy it here): launch.json (mode, command line, source
                                 # commit), config.json (the config as resolved), the rendered manifests; each file
                                 # created once; environment variables by name only, never a value
  plan.json                      # units, costs, estimate
  queue/{todo,claimed,done,failed}/<rank>__<unit>.json   # state; done records name the accepted attempt
  accepted/<unit>.json           # local, ray: the unit's accepted attempt (queue/done and queue/failed are views)
  state/attempts/<unit>/<n>.{begin,end}.json   # local, ray: each attempt's start (its rows) and how it ended (the rows
                                               # it finished, the rows it is charged for)
  state/rows/<unit>/<run>/<row>.json           # local, ray: the attempt each row was accepted from, created once
  units/<unit id>/<attempt>.outcome.json       # local, ray: what the simulator's slot recorded at the end
  k8s/<launch id>.finished       # ray on k8s: the head's end of the launch (its workers then leave)
  k8s/<launch id>.release/<pod>  # ray on k8s: a worker pod its driver let go (that pod then ends)
  units/<unit id>/<attempt>/     # one run_benchmark_cell.py output per attempt (arena/, rows/, run_manifest.json)
  units/<unit id>/<attempt>.log  # that attempt's simulator log
  logs/<worker>/server-<port>-<worker start>-<n>.log   # each server start (<worker>: the station, s<i>, for local, ray)
  logs/s<i>/server-process.log, logs/s<i>/slot<k>.log  # local: a station's server process and its slots' processes
  logs/local<gpu>.log            # a local launch from before the driver: each worker's log
  score_<task>.json              # --score
  retries_<task>.json            # --score: the accepted units that had a failed attempt first, each failure's cause
  score_<task>.accepted.json     # --score: what the scorer reads of each attempt's directory (its accepted rows)
```

**Retried units in a score.** A unit can fail one attempt and pass the next. Some causes are the launcher's (a lost
pod, a hung simulator), but some would fail the episode itself: a policy that drives the simulator to a non-finite
state ends the run, and the retry, on a GPU where Newton is not deterministic, may not. Until the cell runner records
such an episode as failed, `--score` prints how many accepted units ran after a failed attempt and each failure's
cause, and writes them to `retries_<task>.json`; read it before reporting a score as a headline.

**Salvaged episodes in a score.** On a driver launch (`local` or `ray`) a unit is accepted row by row: episodes completed before a
simulator crash are scored, the crashed episode's row is rerun from its initial state, and the attempt limit
(`max_attempts`) is per row. A unit's accepted record lists, in `parts`, each attempt it is read from with its rows,
and `--score` passes those rows to the scorer (`score_<task>.accepted.json`), which reads a crashed attempt's episodes
from Arena's per-episode result lines (it has no experiment result) and scores each row once. The score reports per
cell `attempts.salvaged_episodes` (from attempts a crash ended) and `attempts.retried_rows` (rows charged in an
earlier attempt), and `--score` prints both per task: state them with the score. The directory backends
(`backend.kind: local` or `k8s` with a queue) still rerun the whole unit.

## Adding tasks or models

- **A task** needs no launcher change: once its spec is in the manifest (with frozen rows and a cap), `tasks: all`
  plans it. To add a task to a launch: `launch.sh my_eval.yaml --start --tasks <task>`. Workers still running pick
  its units up; if they have all exited, resume the launch to start them again.
- **A model** is its config's `model` block plus a server recipe: a registered model by its key (an entry in
  `benchmark/models.py`; the headline models are), or a model of your own by its key, label, group and client
  policy block, with no change to the repo. Each run manifest records the whole entry, and `--score` reports a
  config model apart from the headline, in the `community` group. If the simulator needs the
  model's client package, list it in `backend.pythonpath`. [CONNECT_YOUR_POLICY.md](CONNECT_YOUR_POLICY.md) walks
  through a model of your own end to end: a server for the HTTP contract (the server kit,
  `source/dexbench_policy_server`), the config's model block, a two-episode check, and
  `examples/own_policy_local.yaml`.

## Known limits

- The `k8s` backend runs one server per GPU (colocated); other layouts need `local` or `ray`.
- `local` and `ray`: a station switches checkpoint only once its simulators are idle. `ray`: `--k8s-job` renders a
  Ray cluster of pods; on SLURM nodes, start it by hand (`ray start`).
- One tree per launch.
- `ray` on k8s: the workers Job sets `podReplacementPolicy: Failed`, which Kubernetes honours by default from 1.29. The spares
  kept for the running units' retries are counted per station, not spread across pods: a lost pod that held a
  running unit and its spare waits for the pod to come back. When the head's Job fails for good (its
  `head_restarts` spent), no driver writes the launch's end, so its worker pods hold their GPUs until the workers
  Job's `deadline_h`: delete that Job by hand.
- Ray: the cluster runs without Ray's TLS or authentication. No secret value goes through Ray (the servers' secrets
  travel by name, above), so neither is needed on a private network; add them when Ray runs on a shared one, where
  anyone who can reach the head could join the cluster or read what the driver sends.
- Cells only: `sweep: true` is refused, and pilot models (`ModelEntry.rows`) are refused; run those with
  `run_benchmark_cell.py`.
- The sentinel holds one GPU through the tail, by design; two workers that replace a dead sentinel at the same moment
  may both stay.
- `--score` refuses until every planned unit is done (`--partial` scores what is).
