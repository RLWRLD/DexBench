# SPDX-License-Identifier: Apache-2.0
"""A rigid object spawned from several usds, one member per environment, each member seated at its own pose."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.utils.pose import Pose

from dexbench_arena.evaluation.replay_guards import deterministic_reset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def assert_sequential_heterogeneous_cloning() -> None:
    """Fail unless heterogeneous prototypes clone round-robin: environment ``e`` gets prototype ``e mod n``."""

    from isaaclab.cloner import CloneCfg, cloner_strategies

    strategy = CloneCfg().clone_strategy
    if strategy is not cloner_strategies.sequential:
        raise RuntimeError(
            f"heterogeneous cloning defaults to {getattr(strategy, '__name__', strategy)!r}, "
            "not sequential: environment e would not get member e mod n, and every seat in "
            "this set assumes it does. Set scene.clone_cfg.clone_strategy = "
            "isaaclab.cloner.cloner_strategies.sequential on the environment's scene cfg."
        )


class SeatedObjectSet(Object):
    """One rigid scene entity from several usds; environment ``e`` holds member ``e mod n`` at ``seats[e mod n]``."""

    def __init__(
        self,
        name: str,
        member_usd_paths: Sequence[str],
        seats: Sequence[Pose],
        spawn_cfg_addon: dict | None = None,
        **kwargs,
    ):
        import isaaclab.sim as sim_utils

        assert len(member_usd_paths) == len(seats) >= 1, "one usd and one seat per member"
        self.member_seats: tuple[Pose, ...] = tuple(seats)
        assert_sequential_heterogeneous_cloning()
        super().__init__(
            name=name,
            object_type=ObjectType.RIGID,
            spawner_cfg=sim_utils.MultiUsdFileCfg(usd_path=list(member_usd_paths), **(spawn_cfg_addon or {})),
            initial_pose=self.member_seats[0],
            **kwargs,
        )

    def _build_reset_event(self) -> EventTermCfg | None:
        if not self._requires_reset_pose_event():
            return None
        return EventTermCfg(
            func=SeatObjectSetMember,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(self.name),
                "seats": tuple((*seat.position_xyz, *seat.rotation_xyzw) for seat in self.member_seats),
            },
        )


@deterministic_reset
class SeatObjectSetMember(ManagerTermBase):
    """Event term: put each resetting environment's member back on its seat, at rest."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        assert hasattr(self.asset, "write_root_pose_to_sim_index"), "a seated object set must be a rigid object"
        self.seats = torch.tensor(cfg.params["seats"], dtype=torch.float32, device=self.asset.device)
        self._defaults_seated = False

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,  # noqa: ARG002
        seats: tuple[tuple[float, ...], ...],  # noqa: ARG002  (read once in __init__)
    ):
        device = self.asset.device
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.int32)
        else:
            env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.int32).reshape(-1)
        if env_ids.numel() == 0:
            return
        if not self._defaults_seated:
            default_root_pose = wp.to_torch(self.asset.data.default_root_pose)
            members = torch.arange(env.scene.num_envs, device=device) % len(self.seats)
            default_root_pose[:] = self.seats[members].to(default_root_pose.device, default_root_pose.dtype)
            self._defaults_seated = True
        env_index = env_ids.long()
        pose = self.seats[env_index % len(self.seats)].clone()
        pose[:, :3] += env.scene.env_origins[env_index]
        self.asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.asset.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(len(env_ids), 6, device=device), env_ids=env_ids
        )
