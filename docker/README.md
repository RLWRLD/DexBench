# DexBench Docker

This guide builds and runs the DexBench simulation image on your machine. Each
user gets the NVIDIA base image and builds the image locally. Run every host
command from the root of the checkout. [Getting started](../docs/Getting_Started.md) covers the
install, the assets, the datasets, evaluation and teleoperation.
[Eval_Protocol.md](../docs/Eval_Protocol.md) covers the benchmark.

## Build

Prerequisites: Docker with NVIDIA GPU support (the NVIDIA container toolkit),
access to the pinned NVIDIA base image (`BASE_IMAGE` in `docker/Dockerfile`), and
an NVIDIA GPU with RT cores (**GPU** in [Getting started](../docs/Getting_Started.md#1-install)).

Initialise only the two source checkouts that the recipe copies. The other nested
checkouts are not necessary:

```bash
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init submodule/isaaclab_arena
GIT_LFS_SKIP_SMUDGE=1 git -C submodule/isaaclab_arena submodule update --init submodules/IsaacLab

python3 docker/build.py
```

The builder needs a clean checkout and the two source checkouts at their pinned
commits. It builds from a Git snapshot of the commit, without untracked or ignored
files and without other nested checkouts. It writes the source, submodule,
assets-tree and CloudXR-profile identifiers into the image labels.
`docker/Dockerfile.dockerignore` keeps the asset payloads, the datasets and the
outputs out of the build. The default tag is `dexbench_arena:external`; `--tag`
selects another local name.

The image carries the committed source and the CloudXR profile
(`/workspaces/dexbench_arena/docker/cloudxr_quest3.env`). Commit a change and
build again before you run it.

After the build, check the image's labels and its installed recording packages:

```bash
python3 scripts/verify_reproducibility.py --check image --image dexbench_arena:external
```

A direct `docker build` leaves the provenance labels at their defaults (`unknown`,
and `false` for `io.dexbench.reproducible`), and this check refuses such an image. Missing pins, a dirty source tree and image labels that do not
agree with the commit are errors. The check identifies the build environment. It
does not validate task success or replay compatibility.

## Run

The example uses the image's source, mounts the assets read-only and keeps the
generated output owned by the host user. Its separate home mount supplies the
directory that the entrypoint needs and keeps the user caches. Restore the asset
payloads that you need on the host first ([Getting started](../docs/Getting_Started.md#2-assets));
their licenses are separate. Set `ACCEPT_EULA=Y` only after you accept the
applicable NVIDIA terms.

```bash
mkdir -p outputs "$HOME/.cache/dexbench_arena/external/home"

docker run --rm -it --gpus all --network host --ipc host \
  -e ACCEPT_EULA=Y \
  -e DOCKER_RUN_USER_NAME="$(id -un)" \
  -e DOCKER_RUN_USER_ID="$(id -u)" \
  -e DOCKER_RUN_GROUP_NAME="$(id -gn)" \
  -e DOCKER_RUN_GROUP_ID="$(id -g)" \
  --mount "type=bind,src=$HOME/.cache/dexbench_arena/external/home,dst=/home/$(id -un)" \
  --mount "type=bind,src=$PWD/assets,dst=/workspaces/dexbench_arena/assets,readonly" \
  --mount "type=bind,src=$PWD/outputs,dst=/workspaces/dexbench_arena/outputs" \
  dexbench_arena:external
```

`outputs/` holds the frozen benchmark rows (`./scripts/download_datasets.sh
--eval-conditions`) and everything that a run writes. The command mounts no
dataset or model directory. Evaluation and teleoperation do not need one. Add a
`--mount` when you need one.

## CloudXR for teleoperation

For headset sessions, create one shared host directory before you start either
container:

```bash
mkdir -p /tmp/dexbench_cloudxr
```

Open two container shells with the run command above. In each command, before
`dexbench_arena:external`, add
`--mount type=bind,src=/tmp/dexbench_cloudxr,dst=/cloudxr`. Give the first container
`--name dexbench-cloudxr` and the second `--name dexbench-simulation`.
Both use the same host UID and GID. Open the necessary firewall ports on the host:
49100/tcp (signaling), 47998/udp (media) and, in HTTPS proxy mode, 48322/tcp.

In the first shell, start the server after you accept its applicable EULA. The
image contains the CloudXR profile, so the profile needs no mount:

```bash
/isaac-sim/python.sh -m isaacteleop.cloudxr \
  --cloudxr-install-dir /cloudxr --accept-eula \
  --cloudxr-env-config=/workspaces/dexbench_arena/docker/cloudxr_quest3.env
```

Keep that server running. When its startup is complete, load the runtime in the
second shell. Do this before you run the OpenXR teleoperation or recording
commands:

```bash
cd /workspaces/dexbench_arena
source /cloudxr/run/cloudxr.env
```

[Getting started](../docs/Getting_Started.md#6-teleoperate-with-a-meta-quest-over-openxr) gives the
teleoperation and recording commands. These commands do not check the image. Run
the image check under [Build](#build), and build again after you commit a source
change. Headset streaming and task recording need a check of their own; the GPU
smoke check does not cover them.

## GPU smoke check

From the container shell opened above, run each backend in a separate process.
Keep the read-only `assets` mount: environment registration reads the asset
metadata files in it, although the physics scene uses only primitive shapes. The
check does not use the robot and scene payloads, but it needs the downloaded
`assets/` ([Getting started](../docs/Getting_Started.md#2-assets)):

```bash
/isaac-sim/python.sh /opt/dexbench-smoke-simulation.py --viz none --enable_cameras --device cuda:0 --presets physx
/isaac-sim/python.sh /opt/dexbench-smoke-simulation.py --viz none --enable_cameras --device cuda:0 --presets newton
```

Each command must exit with zero and print `SIMULATION_GPU_SMOKE=` followed by
JSON. The check starts Kit with RTX enabled, does a CUDA matrix multiplication,
registers the DexBench environments, and steps a primitive box onto a static
floor for 120 steps. It checks that the CUDA physics state is finite and that the
contact settles. It downloads no robot or scene asset. The first RTX launch can
spend several minutes to compile shaders. This is a runtime smoke check: it does
not exercise benchmark tasks, policy inference or camera pixel output.

The inherited upstream package set is not dependency-clean: `pip check` reports
missing optional packages and version conflicts, including OpenPI's NumPy bound.
The smoke check does not show that every training, data conversion or policy
integration works. Examine `/isaac-sim/python.sh -m pip check` when you use those
optional paths. A successful image build does not validate them.

## Dependencies and policy integrations

The recipe installs the pinned upstream package layout that Arena's runtime
imports need. Optional policy servers, training environments and model weights
need their own setup and applicable terms. Source files for optional integrations
can be present without their dependencies. [Reference checkpoints](../docs/Getting_Started.md#reference-checkpoints)
lists the serving code of each reference model. The benchmark model list and the
scoring rules still apply: a local simulation check is not a complete benchmark
result.

### Optional GR00T remote-policy client

For the GR00T benchmark variants, install the remote-policy client in the local
container. Then run the model server in its separate upstream environment. This
optional setup uses NVIDIA Isaac-GR00T `n1.7-release` at
`23ace64f17aa5015259b8609d371eb61a357c776`, whose source
[LICENSE is Apache-2.0](https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/LICENSE).
Model weights and the server's dependencies have their own terms.

From the container shell, install the pinned source and its compatible
Transformers version. Do not resolve the training dependencies:

```bash
sudo -H env GIT_LFS_SKIP_SMUDGE=1 /isaac-sim/python.sh -m pip install \
  --no-deps --ignore-requires-python \
  transformers==4.57.3 \
  'gr00t @ git+https://github.com/NVIDIA/Isaac-GR00T@23ace64f17aa5015259b8609d371eb61a357c776'

/isaac-sim/python.sh - <<'PY'
from gr00t.policy.server_client import PolicyClient
import dexbench_arena_gr00t.policy.wuji2_gr00t_modality
print("GR00T client and DexBench modality imports passed")
PY
```

Upstream declares Python `==3.10.*`, and this simulator uses Python 3.12.
`--ignore-requires-python` is a deliberate exception for the remote client;
`--no-deps` alone is rejected. An import of `gr00t.policy.server_client` also
loads upstream's policy package initializer, so Transformers must be compatible
with the recipe's Hugging Face Hub version. The explicit `4.57.3` pin matches this
upstream release and does not rely on an inherited Transformers version.

This replaces Transformers for the whole container. Isaac Lab's
`isaaclab.envs.mdp.observations.image_features` also imports Transformers when it
loads a Theia encoder (`_prepare_theia_transformer_model`). The client and GPU
smoke checks do not exercise that optional encoder path. If you use Theia,
validate it separately: the smoke results do not show that it is compatible with
the client installation.

The Python 3.12 checks cover the client and modality imports, a local ZeroMQ ping,
NumPy action round trips, modality serialization and reset against a mock policy
server. They do not validate model inference or training on Python 3.12. Use
upstream's Python 3.10 environment for the model server and training, and check an
actual checkpoint separately before benchmark evaluation. The optional client
changes this container's packages: run the GPU smoke checks again after you
install it. The default image recipe does not install it.

## Licenses

Isaac Sim/Kit, IsaacTeleop/CloudXR and the other installed dependencies keep
their own licenses. This recipe does not turn the runtime into an Apache-only
bundle, and it does not give permission to redistribute a locally built image.
See the
[Isaac Sim license FAQ](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/common/license-faq.html).

[LICENSE](../LICENSE) and [NOTICE](../NOTICE) describe the source-code license
scope. Runtime dependencies, assets, datasets and model weights keep their own
terms.
