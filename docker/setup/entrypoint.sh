#!/bin/bash
set -euo pipefail

# Create a container user matching the host user so generated files keep the
# expected ownership on mounted volumes.

ldconfig

userdel "$DOCKER_RUN_USER_NAME" 2>/dev/null || true
userdel ubuntu 2>/dev/null || true

groupadd --force --gid "$DOCKER_RUN_GROUP_ID" "$DOCKER_RUN_GROUP_NAME"

useradd --no-log-init \
        --uid "$DOCKER_RUN_USER_ID" \
        --gid "$DOCKER_RUN_GROUP_NAME" \
        --groups sudo,isaac-sim \
        --shell /bin/bash \
        "$DOCKER_RUN_USER_NAME"

chown "$DOCKER_RUN_USER_NAME:$DOCKER_RUN_GROUP_NAME" "/home/$DOCKER_RUN_USER_NAME"
for path in "${WORKDIR:-}" "${ARENA_WORKDIR:-}" "${DEXBENCH_WORKDIR:-}"; do
    if [ -n "$path" ] && [ -d "$path" ]; then
        chown "$DOCKER_RUN_USER_NAME:$DOCKER_RUN_GROUP_NAME" "$path"
    fi
done

echo 'root:root' | chpasswd
echo "$DOCKER_RUN_USER_NAME:root" | chpasswd

echo "$DOCKER_RUN_USER_NAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
touch "/home/$DOCKER_RUN_USER_NAME/.sudo_as_admin_successful"

cp /etc/bash.bashrc "/home/$DOCKER_RUN_USER_NAME/.bashrc"
chown "$DOCKER_RUN_USER_NAME:$DOCKER_RUN_GROUP_NAME" "/home/$DOCKER_RUN_USER_NAME/.bashrc"

# The CloudXR mount (/cloudxr, from CXR_HOST_VOLUME_PATH) is the writable state
# dir where isaacteleop.cloudxr creates logs/ and run/ and copies its bundled
# runtime into. Docker creates a missing bind-mount source as root, so hand it
# to the container user like the other host mounts above; otherwise the teleop
# launcher fails with EACCES on mkdir /cloudxr/logs. It is always small (a few
# copied .so's plus logs), so a recursive chown is cheap.
if [ -d /cloudxr ]; then
    chown -R "$DOCKER_RUN_USER_NAME:$DOCKER_RUN_GROUP_NAME" /cloudxr
fi

# Isaac Lab caches remote Nucleus assets under $TMPDIR/Assets (see
# isaaclab.utils.assets.retrieve_file_path), creating directories with the
# invoking uid. A root-run process (e.g. a raw `docker exec` without --user)
# can leave root-owned intermediate dirs that later fail the container user's
# downloads with EACCES. Own the cache root up front and self-heal stale
# root-owned entries on every container start; the cache is bounded (a few
# hundred MB of props at most), so the recursive chown is cheap.
mkdir -p /tmp/Assets
chown -R "$DOCKER_RUN_USER_NAME:$DOCKER_RUN_GROUP_NAME" /tmp/Assets

if [ -n "${ISAACLAB_PATH:-}" ] && [ ! -e "$ISAACLAB_PATH/_isaac_sim" ]; then
    ln -s /isaac-sim/ "$ISAACLAB_PATH/_isaac_sim"
fi

if [ $# -ge 1 ]; then
    exec sudo --preserve-env -u "$DOCKER_RUN_USER_NAME" \
        -- env HOME="/home/$DOCKER_RUN_USER_NAME" bash -ic "$@"
else
    exec su "$DOCKER_RUN_USER_NAME"
fi
