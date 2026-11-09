# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""An alias of :mod:`dexbench_arena_gr00t.policy.wuji2_gr00t_config`."""

import runpy

if __name__ == "__main__":
    runpy.run_module("dexbench_arena_gr00t.policy.wuji2_gr00t_config", run_name="__main__", alter_sys=True)
else:
    from dexbench_arena_gr00t.policy.wuji2_gr00t_config import write_config  # noqa: F401
