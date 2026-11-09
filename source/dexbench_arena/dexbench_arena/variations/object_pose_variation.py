# SPDX-License-Identifier: Apache-2.0
"""Per-env scene-asset placement variation."""

from __future__ import annotations

import math
import torch
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import warp as wp
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass

from isaaclab_arena.utils.yaw import rotate_points_by_yaw
from isaaclab_arena.variations.continuous_sampler import ContinuousSampler
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class PoseDomain(Protocol):
    """What a pose variation's ``domain`` provides: the scene's hard limit on its draws."""

    reason: str
    """What the limit protects, for the error message."""

    def violation(self, low: Sequence[float], high: Sequence[float]) -> str | None:
        """Why a ``[dx, dy, dyaw]`` range ``[low, high]`` can leave the domain, or None when it cannot."""
        ...


@dataclass(frozen=True)
class PlanarPoseDomain:
    """Hard physical limits on where a ``[dx, dy, dyaw]`` draw may put its asset."""

    max_abs_xy_m: tuple[float, float] | None = None
    """Largest ``|dx|`` and ``|dy|`` a draw may reach at the default orientation, each inclusive. None: unbounded."""

    footprint_half_extents_xy_m: tuple[float, float] | None = None
    """The asset's half-extents along world X and Y at its default pose, centred on its origin."""

    max_xy_radius_m: float | None = None
    """Planar distance every draw must stay strictly inside. None: unbounded."""

    reason: str = ""
    """What the limit protects, for the error message."""

    def violation(self, low: Sequence[float], high: Sequence[float]) -> str | None:
        """Why the range ``[low, high]`` can leave the domain, or None when every draw stays in it."""

        reach_x = max(abs(float(low[0])), abs(float(high[0])))
        reach_y = max(abs(float(low[1])), abs(float(high[1])))
        if self.max_abs_xy_m is not None:
            cap_x, cap_y = self.max_abs_xy_m
            turned = ""
            reach_yaw = max(abs(float(low[2])), abs(float(high[2])))
            if self.footprint_half_extents_xy_m is not None and reach_yaw > 0.0:
                grow_x, grow_y = self._turned_growth(reach_yaw)
                cap_x, cap_y = cap_x - grow_x, cap_y - grow_y
                turned = f" once a turn of up to {math.degrees(reach_yaw):.1f} deg widens the footprint"
            if reach_x > cap_x or reach_y > cap_y:
                return (
                    f"the draw reaches |dx| {reach_x * 1000:.0f} mm, |dy| {reach_y * 1000:.0f} mm, past the "
                    f"limit {cap_x * 1000:.0f} mm, {cap_y * 1000:.0f} mm{turned}"
                )
        if self.max_xy_radius_m is not None:
            reach = math.hypot(reach_x, reach_y)
            if reach >= self.max_xy_radius_m:
                return (
                    f"a draw of {reach * 1000:.0f} mm reaches or passes the radius "
                    f"{self.max_xy_radius_m * 1000:.0f} mm"
                )
        return None

    def _turned_growth(self, reach_yaw: float) -> tuple[float, float]:
        """How much a turn of up to ``reach_yaw`` widens the footprint along X and Y (the worst turn in range)."""
        half_x, half_y = self.footprint_half_extents_xy_m
        corners = torch.tensor([[half_x, half_y, 0.0], [half_x, -half_y, 0.0]], dtype=torch.float64)
        turns = {reach_yaw} | {turn for turn in (math.atan2(half_y, half_x), math.atan2(half_x, half_y)) if turn < reach_yaw}
        widest = torch.stack([rotate_points_by_yaw(corners, turn)[:, :2].abs().amax(dim=0) for turn in turns]).amax(dim=0)
        return max(float(widest[0]) - half_x, 0.0), max(float(widest[1]) - half_y, 0.0)


@dataclass(frozen=True)
class SupportedPlanarPoseDomain:
    """Keep an off-centre rectangular footprint entirely on a rectangular support."""

    default_xy_m: tuple[float, float]
    default_yaw_rad: float
    local_bounds_xy_m: tuple[tuple[float, float], tuple[float, float]]
    support_bounds_xy_m: tuple[tuple[float, float], tuple[float, float]]
    reason: str = "the complete footprint must remain on its support"

    def violation(self, low: Sequence[float], high: Sequence[float]) -> str | None:
        if len(low) != 3 or len(high) != 3 or not all(math.isfinite(float(v)) for v in (*low, *high)):
            return "the planar draw must contain three finite bounds per endpoint"
        if any(lo > hi for lo, hi in zip(low, high)):
            return "the planar draw has a lower bound above its upper bound"
        yaw_low = self.default_yaw_rad + float(low[2])
        yaw_high = self.default_yaw_rad + float(high[2])
        if yaw_high - yaw_low >= 2 * math.pi:
            yaw_low, yaw_high = -math.pi, math.pi
        corners = torch.tensor(
            [[x, y, 0.0] for x in self.local_bounds_xy_m[0] for y in self.local_bounds_xy_m[1]],
            dtype=torch.float64,
        )
        turns = {yaw_low, yaw_high}
        for x, y, _ in corners.tolist():
            phase = math.atan2(y, x)
            first = math.ceil((yaw_low + phase) / (math.pi / 2))
            last = math.floor((yaw_high + phase) / (math.pi / 2))
            turns.update(k * math.pi / 2 - phase for k in range(first, last + 1))
        turned = torch.cat([rotate_points_by_yaw(corners, yaw)[:, :2] for yaw in turns])
        lower = turned.amin(dim=0) + torch.tensor([p + d for p, d in zip(self.default_xy_m, low[:2])])
        upper = turned.amax(dim=0) + torch.tensor([p + d for p, d in zip(self.default_xy_m, high[:2])])
        for axis, (minimum, maximum) in enumerate(self.support_bounds_xy_m):
            if float(lower[axis]) < minimum - 1e-7 or float(upper[axis]) > maximum + 1e-7:
                return (
                    f"the {'XY'[axis]} footprint reaches [{float(lower[axis]):.4f}, {float(upper[axis]):.4f}] m "
                    f"outside the support [{minimum:.4f}, {maximum:.4f}] m"
                )
        return None


class PlanarOffsetListener:
    """The last ``(dx, dy)`` a planar-pose variation drew, per environment, for a term that follows the moved asset."""

    def __init__(self):
        self._xy = torch.zeros(0, 2)

    def __call__(self, sample, env_ids) -> None:
        if env_ids is None:
            return
        index = torch.as_tensor(env_ids, dtype=torch.long).cpu()
        if len(index) and int(index.max()) >= len(self._xy):
            grown = torch.zeros(int(index.max()) + 1, 2)
            grown[: len(self._xy)] = self._xy
            self._xy = grown
        self._xy[index] = torch.as_tensor(sample, dtype=torch.float32).cpu()[:, :2]

    def __repr__(self) -> str:
        return "PlanarOffsetListener()"

    def __deepcopy__(self, memo):
        return self

    def xy(self, num_envs: int, device, dtype) -> torch.Tensor:
        """``(num_envs, 2)`` offsets on ``device``; environments not drawn yet read zero."""
        xy = torch.zeros(num_envs, 2)
        count = min(num_envs, len(self._xy))
        xy[:count] = self._xy[:count]
        return xy.to(device=device, dtype=dtype)


@configclass
class ObjectPoseVariationCfg(VariationBaseCfg):
    """Configuration for ObjectPoseVariation."""

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(low=[-0.05, -0.05, -0.35], high=[0.05, 0.05, 0.35])
    )
    """Uniform ``[dx, dy, dyaw]``: env-frame metres, and radians about world +Z (a turn in the table plane)."""


class ObjectPoseVariation(RunTimeVariationBase):
    """Vary a rigid object or articulation root -- or a group of them -- at reset."""

    cfg: ObjectPoseVariationCfg

    _replaying = False
    """Recorded draws are bound (Arena's ``set_replay_sampler``): the range draws nothing."""

    _recorded_draws_checked = False
    """:meth:`check_recorded_draws` has held the rows to :attr:`domain` since the last unbinding."""

    def __init__(
        self,
        asset_name: str,
        cfg: ObjectPoseVariationCfg | None = None,
        name: str = "pose",
        domain: PoseDomain | None = None,
        member_names: Sequence[str] | None = None,
        on_applied: Callable[[ManagerBasedEnv, torch.Tensor, torch.Tensor], None] | None = None,
    ):
        super().__init__(cfg=cfg if cfg is not None else ObjectPoseVariationCfg(), name=name)
        self.asset_name = asset_name
        self.domain = domain
        self.on_applied = on_applied
        self.member_names = tuple(member_names) if member_names is not None else (asset_name,)
        if not self.member_names:
            raise ValueError(f"{type(self).__name__} on '{asset_name}': member_names is empty")

        self.add_sample_listener(self._validate_samples)

    def _validate_samples(self, samples, env_ids) -> None:
        if self.domain is None:
            return
        for sample in torch.as_tensor(samples).reshape(-1, 3).tolist():
            problem = self.domain.violation(sample, sample)
            if problem is not None:
                raise ValueError(
                    f"{type(self).__name__} '{self.asset_name}.{self.name}': {problem} "
                    f"({self.domain.reason}); refusing draw {sample}"
                )

    def set_replay_sampler(self, replay_sampler) -> None:
        """Arena's binding of recorded draws (``None`` unbinds), noted so the build knows which check applies."""
        super().set_replay_sampler(replay_sampler)
        self._replaying = replay_sampler is not None
        if replay_sampler is None:
            self._recorded_draws_checked = False

    def check_recorded_draws(self, draws: Sequence[tuple[str, object]]) -> None:
        """Refuse (``ValueError``) a recorded draw that leaves :attr:`domain`; ``draws`` is ``(row label, draw)``
        for every row the build will replay. Once they pass, a build with them bound checks them in place of the
        sampler range (the module docstring)."""
        for label, draw in draws:
            values = torch.as_tensor(draw, dtype=torch.float64).reshape(-1).tolist() if draw is not None else None
            if values is None or len(values) != 3:
                raise ValueError(f"{type(self).__name__} '{self.asset_name}.{self.name}': {label} records {draw!r}, "
                                 "not one [dx, dy, dyaw]")
            problem = self.domain.violation(values, values) if self.domain is not None else None
            if problem is not None:
                raise ValueError(
                    f"{type(self).__name__} '{self.asset_name}.{self.name}': {label} records {values}: {problem}"
                    f"{f' ({self.domain.reason})' if self.domain.reason else ''}; refusing the replay"
                )
        self._recorded_draws_checked = True

    def validate_range(self) -> None:
        """Raise if the current sampler range can put the asset outside :attr:`domain`."""

        if self.domain is None:
            return
        problem = self.domain.violation(self.cfg.sampler_cfg.low, self.cfg.sampler_cfg.high)
        if problem is not None:
            raise ValueError(
                f"{type(self).__name__} '{self.asset_name}.{self.name}': {problem}"
                f"{f' ({self.domain.reason})' if self.domain.reason else ''}. "
                f"Sampler range low={list(self.cfg.sampler_cfg.low)}, high={list(self.cfg.sampler_cfg.high)}; "
                "shrink it."
            )

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        assert self._sampler is not None, (
            f"ObjectPoseVariation on '{self.asset_name}' is enabled but no sampler is set; "
            "call apply_cfg with a cfg that sets sampler_cfg before building the env."
        )
        if not (self._replaying and self._recorded_draws_checked):
            self.validate_range()
        event_cfg = EventTermCfg(
            func=ApplyObjectPoseFromSampler,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(self.asset_name),
                "member_names": list(self.member_names),
                "sampler": self._sampler,
                **({"on_applied": self.on_applied} if self.on_applied is not None else {}),
            },
        )
        return f"{self.asset_name}_{self.name}_variation", event_cfg


def make_planar_pose_variation(
    asset_name: str,
    *,
    xy_half_extent_m: float = 0.0,
    yaw_half_extent_rad: float = 0.0,
    xy_offset_m: Sequence[float] = (0.0, 0.0),
    yaw_offset_rad: float = 0.0,
    name: str = "pose",
    domain: PoseDomain | None = None,
    member_names: Sequence[str] | None = None,
    low: Sequence[float] | None = None,
    high: Sequence[float] | None = None,
) -> ObjectPoseVariation:
    """Create a recorded planar-pose variation for any scene asset, symmetric unless bounded."""

    xy_half_extent_m = float(xy_half_extent_m)
    yaw_half_extent_rad = float(yaw_half_extent_rad)
    if xy_half_extent_m < 0.0:
        raise ValueError("Planar pose XY half-extent must be non-negative")
    if yaw_half_extent_rad < 0.0:
        raise ValueError("Planar pose yaw half-extent must be non-negative")
    if len(xy_offset_m) != 2 or not all(math.isfinite(float(value)) for value in xy_offset_m):
        raise ValueError(f"Planar pose XY offset must be two finite values, got {tuple(xy_offset_m)}")
    offset_x, offset_y = (float(value) for value in xy_offset_m)
    offset_yaw = float(yaw_offset_rad)
    if not math.isfinite(offset_yaw):
        raise ValueError(f"Planar pose yaw offset must be finite, got {yaw_offset_rad}")
    if (low is None) != (high is None):
        raise ValueError("Planar pose bounds take both low and high")
    if low is None:
        low = [-xy_half_extent_m, -xy_half_extent_m, -yaw_half_extent_rad]
        high = [xy_half_extent_m, xy_half_extent_m, yaw_half_extent_rad]
    elif xy_half_extent_m or yaw_half_extent_rad:
        raise ValueError("Planar pose bounds replace the half-extents; give one or the other")
    low, high = [float(v) for v in low], [float(v) for v in high]
    if len(low) != 3 or len(high) != 3 or any(lo > hi for lo, hi in zip(low, high)):
        raise ValueError(f"Planar pose bounds are [dx, dy, dyaw] with low <= high; got {low}, {high}")
    low[0], high[0], low[1], high[1] = low[0] + offset_x, high[0] + offset_x, low[1] + offset_y, high[1] + offset_y
    low[2], high[2] = low[2] + offset_yaw, high[2] + offset_yaw

    return ObjectPoseVariation(
        asset_name,
        name=name,
        domain=domain,
        member_names=member_names,
        cfg=ObjectPoseVariationCfg(
            enabled=any(low) or any(high),
            sampler_cfg=UniformSamplerCfg(low=low, high=high),
        ),
    )


class ApplyObjectPoseFromSampler(ManagerTermBase):
    """Event term: offset every member's root pose by one sampler draw."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        sampler: ContinuousSampler = cfg.params["sampler"]
        self.members = [env.scene[name] for name in cfg.params["member_names"]]
        for name, member in zip(cfg.params["member_names"], self.members):
            assert hasattr(member, "write_root_pose_to_sim_index"), (
                "ApplyObjectPoseFromSampler expects assets with indexed root-pose setters at "
                f"scene['{name}']; got {type(member).__name__}."
            )
        assert tuple(sampler.shape_per_sample) == (3,), (
            "ApplyObjectPoseFromSampler expects a sampler with shape_per_sample (3,) over [dx, dy, dyaw]; "
            f"got {tuple(sampler.shape_per_sample)}."
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,  # noqa: ARG002
        member_names: list[str],  # noqa: ARG002
        sampler: ContinuousSampler,
        on_applied: Callable[[ManagerBasedEnv, torch.Tensor, torch.Tensor], None] | None = None,
    ):
        device = self.members[0].device
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.int32)
        else:
            env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.int32).reshape(-1)
        if env_ids.numel() == 0:
            return

        sample = sampler.sample(num_samples=len(env_ids), env_ids=env_ids).to(device=device, dtype=torch.float32)
        index = env_ids.long()
        origins = env.scene.env_origins[index]
        zeros = torch.zeros_like(sample[:, 2])
        yaw_delta = math_utils.quat_from_euler_xyz(zeros, zeros, sample[:, 2])

        for member in self.members:
            default_root_pose = wp.to_torch(member.data.default_root_pose)[index].clone()
            default_root_vel = wp.to_torch(member.data.default_root_vel)[index].clone()
            positions = default_root_pose[:, 0:3] + origins
            positions[:, 0] += sample[:, 0]
            positions[:, 1] += sample[:, 1]
            orientations = math_utils.quat_mul(yaw_delta, default_root_pose[:, 3:7])
            member.write_root_pose_to_sim_index(
                root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids
            )
            member.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros_like(default_root_vel), env_ids=env_ids
            )
        if on_applied is not None:
            on_applied(env, env_ids, sample)
