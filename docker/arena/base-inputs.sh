#!/usr/bin/env bash
# Prints the paths the image build consumes, one per line: everything
# docker/Dockerfile COPYs or ADDs, the recipe itself, the .dockerignore that
# shapes its context, and the dependency declarations the editable install
# reads. Derived from the Dockerfile rather than listed by hand, so a COPY
# added later is covered from the commit that adds it. Paths under submodule/
# are left out: git diff never looks below a gitlink, and image-paths.sh names
# submodule/isaaclab_arena itself.
#
#   docker/arena/base-inputs.sh
set -euo pipefail
# No globbing: a COPY source like docker/setup/*.sh is printed as written,
# for git to match, not expanded to this checkout's files.
set -f
dockerfile=docker/Dockerfile
ignore=docker/Dockerfile.dockerignore

# Join continuation lines; keep the sources of each COPY/ADD (all but the last
# word, flags dropped). A form this cannot read fails rather than guarding less.
sources=$(sed -e ':a' -e '/\\$/N; s/\\\n/ /; ta' "$dockerfile" | awk '
  toupper($1) != "COPY" && toupper($1) != "ADD" { next }
  $2 ~ /^\[/ || $0 ~ /--from/ { print "unsupported: " $0 > "/dev/stderr"; exit 1 }
  { for (i = 2; i < NF; i++) if ($i !~ /^--/) {
      if ($i ~ /\$/) { print "unsupported source: " $i > "/dev/stderr"; exit 1 }
      print $i } }')

printf '%s\n' "$dockerfile" "$ignore" source/dexbench_arena/pyproject.toml source/dexbench_arena/setup.py $sources \
  | sed 's:/*$::' | awk '!/^submodule\//' | sort -u
