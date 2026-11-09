# Local-build recipe for external users. See docker/README.md.
ARG BASE_IMAGE=nvcr.io/nvidia/isaac-sim@sha256:af1d2b4e75d553bfa27beb5a401198654aa8d607f3b7a6749196e9ce253def20

FROM ${BASE_IMAGE}

USER root

# The base image deduplicates identical files across /isaac-sim with relative
# symlinks, many of them into pip_prebundle directories. Every pip install
# below can uninstall a package from a prebundle directory on sys.path, which
# deletes the link targets and leaves other extensions (e.g. omni.warp.core)
# pointing at nothing. Copy those targets into place before any pip step.
RUN /bin/bash -euc ' \
    links=$(mktemp); \
    find /isaac-sim -type l -print0 > "$links"; \
    while IFS= read -r -d "" link; do \
      if ! target=$(readlink -e -- "$link"); then \
        printf "Broken symlink in the base image: %s\n" "$link" >&2; exit 1; \
      fi; \
      case "$target" in \
        */pip_prebundle/*) \
          if [ ! -f "$target" ]; then \
            printf "Expected a regular-file target: %s -> %s\n" "$link" "$target" >&2; exit 1; \
          fi; \
          cp -a --remove-destination -- "$target" "$link" ;; \
      esac; \
    done < "$links"; \
    rm -- "$links"'

ARG ARENA_WORKDIR="/workspaces/isaaclab_arena"
ARG DEXBENCH_WORKDIR="/workspaces/dexbench_arena"

ENV ARENA_WORKDIR=${ARENA_WORKDIR}
ENV DEXBENCH_WORKDIR=${DEXBENCH_WORKDIR}
ENV WORKDIR=${ARENA_WORKDIR}
ENV ISAACLAB_PATH=${ARENA_WORKDIR}/submodules/IsaacLab
ENV ISAACLAB_ARENA_PATH=${ARENA_WORKDIR}
ENV DEXBENCH_ARENA_PATH=${DEXBENCH_WORKDIR}
ENV TERM=xterm

WORKDIR "${ARENA_WORKDIR}"

# Hide conflicting Vulkan files, if needed.
RUN if [ -e "/usr/share/vulkan" ] && [ -e "/etc/vulkan" ]; then \
      mv /usr/share/vulkan /usr/share/vulkan_hidden; \
    fi

RUN apt-get update && apt-get install -y \
  git \
  git-lfs \
  cmake \
  ffmpeg \
  sudo \
  jq \
  python3-pip

################################
# Install Isaac Lab
################################
COPY submodule/isaaclab_arena/submodules/IsaacLab ${ARENA_WORKDIR}/submodules/IsaacLab

RUN ln -s /isaac-sim/ ${ISAACLAB_PATH}/_isaac_sim
RUN for DIR in ${ISAACLAB_PATH}/source/isaaclab*/; do /isaac-sim/python.sh -m pip install --no-deps -e "$DIR"; done
RUN chmod 777 -R /isaac-sim/kit/
RUN chmod a+x /isaac-sim
RUN /isaac-sim/python.sh -m pip install --no-deps -e ${ISAACLAB_PATH}/source/isaaclab_visualizers
RUN /isaac-sim/python.sh -m pip install --no-deps -e ${ISAACLAB_PATH}/source/isaaclab_teleop
RUN ${ISAACLAB_PATH}/isaaclab.sh -i

# DAQP version required for latest qpsolvers 4.12.0. Backport of IsaacLab#5556.
RUN /isaac-sim/python.sh -m pip install --force-reinstall daqp==0.8.5

# Isaac Teleop Python APIs (retargeters, device I/O, OpenXR bindings).
# Arena imports isaacteleop directly and pins ==1.4.126rc1 in its own
# dependencies, so the editable arena install further down installs that version
# whatever this layer puts in. Match it here rather than being silently upgraded
# later. Keep this pin aligned with the development recipe.
# isaaclab_teleop at the pinned IsaacLab depends only on isaaclab, so it adds
# no competing bound.
RUN /isaac-sim/python.sh -m pip install 'isaacteleop[retargeters,ui,cloudxr]==1.4.126rc1' --extra-index-url https://pypi.nvidia.com

# The newton stack is pinned by `isaaclab.sh -i` above: its _ensure_newton() force-
# installs the newton git build, newton-usd-schemas, mujoco and mujoco-warp named in the
# RLWRLD/IsaacLab fork's [tool.uv].override-dependencies (Newton main 646cece4, version
# 1.8.0.dev0, the parent of newton#4227; schemas 0.5.0; MuJoCo 3.15.0), which plain pip
# installs do not see; warp-lang 1.17.0 is one of that fork's core dependencies.
RUN /isaac-sim/python.sh -m pip install --upgrade pip==26.1.2
RUN /isaac-sim/python.sh -m pip install av==17.1.0 huggingface-hub==0.36.2 && \
    /isaac-sim/python.sh -m pip install --no-deps \
      datasets==3.6.0 \
      loguru==0.7.3

# Shared transports are also used by DexBench's independent policy clients.
RUN /isaac-sim/python.sh -m pip install msgpack==1.1.0 msgpack-numpy==0.4.8 pyzmq==27.0.1

# The eval launcher's Ray backend (scripts/evaluation/eval_launch/ray_backend.py): its driver and every Ray worker run
# under Isaac Sim's Python, which already has every dependency Ray 2.58.1 asks for, so pip adds only Ray itself.
RUN /isaac-sim/python.sh -m pip install ray==2.58.1

COPY submodule/isaaclab_arena/isaaclab_arena_openpi/docker/OPENPI_COMMIT /tmp/openpi_commit
RUN OPENPI_COMMIT=$(tr -d '[:space:]' < /tmp/openpi_commit) && \
    /isaac-sim/python.sh -m pip install --no-cache-dir \
    "openpi-client @ git+https://github.com/Physical-Intelligence/openpi@${OPENPI_COMMIT}#subdirectory=packages/openpi-client" && \
    rm /tmp/openpi_commit

################################
# Install IsaacLab-Arena
################################
# Preserve the pinned upstream package layout for runtime imports.
COPY submodule/isaaclab_arena/pyproject.toml \
     submodule/isaaclab_arena/extension.toml \
     submodule/isaaclab_arena/README.md \
     submodule/isaaclab_arena/LICENSE.md \
     ${ARENA_WORKDIR}/
COPY submodule/isaaclab_arena/isaaclab_arena ${ARENA_WORKDIR}/isaaclab_arena
COPY submodule/isaaclab_arena/isaaclab_arena_cosmos ${ARENA_WORKDIR}/isaaclab_arena_cosmos
COPY submodule/isaaclab_arena/isaaclab_arena_curobo ${ARENA_WORKDIR}/isaaclab_arena_curobo
COPY submodule/isaaclab_arena/isaaclab_arena_dreamzero ${ARENA_WORKDIR}/isaaclab_arena_dreamzero
COPY submodule/isaaclab_arena/isaaclab_arena_g1 ${ARENA_WORKDIR}/isaaclab_arena_g1
COPY submodule/isaaclab_arena/isaaclab_arena_gr00t ${ARENA_WORKDIR}/isaaclab_arena_gr00t
COPY submodule/isaaclab_arena/isaaclab_arena_openpi ${ARENA_WORKDIR}/isaaclab_arena_openpi
COPY submodule/isaaclab_arena/isaaclab_arena_environments ${ARENA_WORKDIR}/isaaclab_arena_environments
COPY submodule/isaaclab_arena/isaaclab_arena_examples ${ARENA_WORKDIR}/isaaclab_arena_examples
COPY submodule/isaaclab_arena/docs ${ARENA_WORKDIR}/docs

RUN /isaac-sim/python.sh -m pip install -e "${ARENA_WORKDIR}/[dev]"

################################
# Install DexBench
################################
WORKDIR "${DEXBENCH_WORKDIR}"

COPY README.md LICENSE NOTICE ${DEXBENCH_WORKDIR}/
# Payloads are excluded by Dockerfile.dockerignore; docker/README.md
# mounts assets read-only and keeps generated output in a separate host folder.
COPY docs ${DEXBENCH_WORKDIR}/docs
COPY scripts ${DEXBENCH_WORKDIR}/scripts
COPY source ${DEXBENCH_WORKDIR}/source
COPY reproducibility ${DEXBENCH_WORKDIR}/reproducibility

# Leave mount points for separately acquired payloads.
RUN mkdir -p ${DEXBENCH_WORKDIR}/assets ${DEXBENCH_WORKDIR}/data

RUN /isaac-sim/python.sh -m pip install -e "${DEXBENCH_WORKDIR}/source/dexbench_arena"

ENV PYTHONPATH=${DEXBENCH_WORKDIR}/source/dexbench_arena:${ARENA_WORKDIR}

RUN echo "alias python='/isaac-sim/python.sh'" >> /etc/bash.bashrc
RUN echo "alias pip3='/isaac-sim/python.sh -m pip'" >> /etc/bash.bashrc
RUN echo "alias pytest='/isaac-sim/python.sh -m pytest'" >> /etc/bash.bashrc
RUN echo "alias debugpy='python -Xfrozen_modules=off -m debugpy --listen localhost:5678 --wait-for-client'" >> /etc/bash.bashrc

RUN echo "PS1='[DexBench] \[\e[0;32m\]~\u \[\e[0;34m\]\w\[\e[0m\] $ '" >> /etc/bash.bashrc
RUN echo "alias ll='ls -alF --color=auto'" >> /etc/bash.bashrc
RUN echo "alias ..='cd ..'" >> /etc/bash.bashrc
RUN cp /etc/bash.bashrc /root/.bashrc

# The entrypoint attaches interactive shells as the host user,
# but Kit persists its user config, shader cache, and logs under /isaac-sim,
# which is root-owned. Without this, non-root sessions fail to save
# user.config.json, cold-boot every launch (~74 s instead of ~10 s), write no
# kit_*.log, and the GUI can exit right after startup.
RUN mkdir -p /isaac-sim/kit/data /isaac-sim/kit/logs /isaac-sim/kit/cache && \
    chmod -R a+rwX /isaac-sim/kit/data /isaac-sim/kit/logs /isaac-sim/kit/cache

COPY docker/setup/entrypoint.sh /entrypoint.sh
COPY docker/smoke_simulation.py /opt/dexbench-smoke-simulation.py
# The CloudXR profile that teleoperation starts the runtime with (its digest is a label below).
COPY docker/cloudxr_quest3.env ${DEXBENCH_WORKDIR}/docker/cloudxr_quest3.env

# Keep the recording package contract in sync with the development recipe.
# The CPU decoder leaves the simulator's torch installation unchanged.
RUN /isaac-sim/python.sh -m pip install --no-cache-dir --no-deps \
      torchcodec==0.11.0 --index-url https://download.pytorch.org/whl/cpu

# Fail the build if any step above left a dangling link under /isaac-sim.
RUN broken=$(find /isaac-sim -xtype l) && \
    if [ -n "$broken" ]; then \
      printf "Broken symlinks under /isaac-sim:\n%s\n" "$broken" >&2; \
      exit 1; \
    fi

# Keep provenance labels at the end: changing only the DexBench source revision
# must not invalidate the expensive base/IsaacLab dependency layers.
ARG DEXBENCH_IMAGE_TAG="external"
ARG DEXBENCH_SOURCE_REV="unknown"
ARG DEXBENCH_HF_ASSETS_REV="unknown"
ARG DEXBENCH_CLOUDXR_PROFILE_SHA256="unknown"
ARG DEXBENCH_REPRODUCIBLE="false"
ARG DEXBENCH_ARENA_REV="unknown"
ARG DEXBENCH_ISAACLAB_REV="unknown"
# Scripts that record which source they ran (rerecord_demos.py) ask git, and
# the workspace above has no .git; this is what they read instead. A development
# container mounts the checkout over the workspace, and git answers there. The
# publish workflow and recording/rerecord_metadata.py read it by this name.
ENV DEXBENCH_IMAGE_REVISION=${DEXBENCH_SOURCE_REV}
# Re-declared without a value: the build stage takes the global default above,
# so the label says which base this recipe was actually built from.
ARG BASE_IMAGE
LABEL org.opencontainers.image.revision="${DEXBENCH_SOURCE_REV}" \
      io.dexbench.recipe="external-v1" \
      io.dexbench.reproducible="${DEXBENCH_REPRODUCIBLE}" \
      io.dexbench.hf-assets-revision="${DEXBENCH_HF_ASSETS_REV}" \
      io.dexbench.cloudxr-profile-sha256="${DEXBENCH_CLOUDXR_PROFILE_SHA256}" \
      io.dexbench.isaaclab-arena-revision="${DEXBENCH_ARENA_REV}" \
      io.dexbench.isaaclab-revision="${DEXBENCH_ISAACLAB_REV}" \
      io.dexbench.base-image="${BASE_IMAGE}"

ENTRYPOINT ["/entrypoint.sh"]
