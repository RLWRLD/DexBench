# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Shared setup for the logistics pack-invoice VBD paper sidecar."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import torch
import warp as wp
from isaaclab.utils.math import quat_apply, quat_from_matrix, quat_inv, quat_mul
from isaaclab_arena.utils.yaw import rotate_points_by_yaw

from .hand_hulls import hand_hull_links, link_collision_hulls
from .hand_spec import PaperHandSpec
from .newton_solver_slot import capture_without_coupler, main_mujoco_solver
from .pack_invoice_scene_randomization import PAPER_PHYSICS_ARGS
from .paper_coupling import ProxyCouplingCfg, install_paper_coupling
from .paper_stack import (
    OpenBoxCfg,
    PaperStackVBD,
    PaperStackVBDCfg,
    StackSiteCfg,
    StaticBoxCollider,
    set_active_paper_stack,
)

ENV0 = "/World/envs/env_0"

_CFG_OVERRIDE: PackInvoiceSidecarCfg | None = None
_ACTIVE_SIDECAR: PackInvoiceSidecar | None = None


def set_pack_invoice_sidecar_cfg_override(cfg: PackInvoiceSidecarCfg | None) -> None:
    """Cfg ``build_pack_invoice_sidecar`` uses when its caller passes none."""

    global _CFG_OVERRIDE
    _CFG_OVERRIDE = cfg


def get_pack_invoice_sidecar_cfg_override() -> PackInvoiceSidecarCfg | None:
    return _CFG_OVERRIDE


SHEET_SIZE_XY = (0.2024, 0.276)
E2_INVOICE_PILE_XY = (4.48, 3.50)
TELEOP_PAPER_OVERRIDES = {
    "contact_mu": 0.2,
    "self_contact_radius": 0.002,
    "self_contact_margin": 0.003,
    "sheet_spacing_m": 0.004,
}
INVOICE_TEXTURE = os.path.join(os.path.dirname(__file__), "textures", "invoice.png")


_RETIRED_PAPER_ENV = (
    "DEXBENCH_PAPER_COUPLING", "DEXBENCH_PAPER_HAND_COLLIDERS", "DEXBENCH_PAPER_EDGE_GRASP",
    "DEXBENCH_PAPER_NUM_SHEETS", "DEXBENCH_PAPER_SUBSTEPS", "DEXBENCH_PAPER_ITERATIONS",
    "DEXBENCH_PAPER_RESOLUTION", "DEXBENCH_PAPER_CONTACT_MU", "DEXBENCH_PAPER_SELF_CONTACT_RADIUS",
    "DEXBENCH_PAPER_CUDA_GRAPH", "DEXBENCH_PAPER_CARTON_SHIFT", "DEXBENCH_PAPER_FEEDBACK_CAP_N",
    "DEXBENCH_PAPER_FEEDBACK_TORQUE_CAP_NM", "DEXBENCH_PAPER_MESH_BEFORE_RENDER",
    "DEXBENCH_PAPER_TEXTURE", "DEXBENCH_PAPER_DEBUG_GEOMETRY",
)


def refuse_retired_paper_env() -> None:
    """Raise if the shell sets a ``_RETIRED_PAPER_ENV`` variable: the run would not be the one its arguments state."""

    stale = [name for name in _RETIRED_PAPER_ENV if os.environ.get(name, "").strip()]
    if stale:
        raise ValueError(f"{', '.join(stale)}: the paper's physics are the E2 environment's --paper_* arguments "
                         "(pack_invoice_scene_randomization.PAPER_PHYSICS_ARGS); unset these")


def sidecar_cfg_from_args(args: Mapping[str, object] | None = None) -> PackInvoiceSidecarCfg:
    """The sidecar the E2 environment's ``--paper_*`` arguments describe (``PAPER_PHYSICS_ARGS`` fills the rest)."""

    unknown = set(args or {}) - set(PAPER_PHYSICS_ARGS)
    if unknown:
        raise TypeError(f"unknown pack_invoice paper argument(s): {sorted(unknown)}")
    a = {**PAPER_PHYSICS_ARGS, **(args or {})}
    coupling = a["paper_coupling"]
    if coupling != "one_way_proxy":
        raise ValueError(f"paper_coupling={coupling!r}: the proxy hand (one_way_proxy) is the only coupling")
    colliders = a["paper_hand_colliders"]
    if colliders != "official_hull":
        raise ValueError(f"paper_hand_colliders={colliders!r}: the hand's own collision hulls (official_hull) are the "
                         "only hand the paper sees")
    for name in ("paper_num_sheets", "paper_substeps", "paper_iterations"):
        if isinstance(a[name], bool) or not isinstance(a[name], int) or a[name] < 1:
            raise ValueError(f"{name}={a[name]!r}: use a positive integer")
    try:
        nx, ny = (int(v) for v in str(a["paper_resolution"]).lower().split("x"))
    except ValueError:
        nx = ny = 0
    if nx < 1 or ny < 1:
        raise ValueError(f"paper_resolution={a['paper_resolution']!r}: use NXxNY with both at least 1, e.g. 4x6")
    for name in ("paper_contact_mu", "paper_self_contact_radius"):
        if isinstance(a[name], bool) or not isinstance(a[name], (int, float)) or not math.isfinite(a[name]):
            raise ValueError(f"{name}={a[name]!r}: use a finite number")
    mu = float(a["paper_contact_mu"])
    if mu < 0.0:
        raise ValueError(f"paper_contact_mu={mu!r}: must not be negative")
    radius = float(a["paper_self_contact_radius"])
    if not 0.0 < radius < 0.02:
        raise ValueError(f"paper_self_contact_radius={radius!r}: use metres in (0, 0.02)")

    cfg = PackInvoiceSidecarCfg(
        num_sheets=a["paper_num_sheets"],
        substeps=a["paper_substeps"],
        iterations=a["paper_iterations"],
    )
    cfg.paper_cfg_overrides = {"contact_mu": mu, "self_contact_radius": radius,
                               "self_contact_margin": 1.5 * radius, "sheet_spacing_m": 2.0 * radius,
                               "resolution": (nx, ny)}
    print(f"[pack_invoice] paper: coupling {coupling}, hand colliders {colliders}, {cfg.num_sheets} sheet(s) per pile, "
          f"grid {nx}x{ny}, mu {mu:g}, sheet-sheet contact radius {radius:g} m, {cfg.substeps} substeps per "
          f"{cfg.substeps_frame_dt:.4f} s x {cfg.iterations} iterations")
    return cfg


_HAND_DERIVED_PAPER_FIELDS = ("kinematic_hulls", "proxy_mirrors")


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


def sidecar_cfg_from_recording(paper_cfg: dict, coupling_cfg: dict | None = None, *,
                               frame_dt: float) -> PackInvoiceSidecarCfg:
    """The sidecar a paper demo HDF5 was recorded with."""

    if coupling_cfg is None:
        raise ValueError("the recording has no coupling_cfg: it moved a kinematic hand (one_way), a mode that is gone")
    if coupling_cfg.get("two_way"):
        raise ValueError("the recording fed the paper's wrench back into the robot (two_way); that mode is gone")

    if paper_cfg.get("kinematic_boxes") or paper_cfg.get("mirror_boxes"):
        raise ValueError("the recording mirrored the hand as boxes; only the hand's own collision hulls are offered")
    cfg = PackInvoiceSidecarCfg()
    known = ProxyCouplingCfg.__dataclass_fields__
    cfg.coupling = ProxyCouplingCfg(**{k: _tuples(v) for k, v in coupling_cfg.items() if k in known})
    sites = tuple(StackSiteCfg(**{k: _tuples(v) for k, v in s.items()}) for s in paper_cfg["sites"])
    cfg.num_sheets = sites[0].num_sheets
    cfg.site_centers_xy = tuple(s.center_xy for s in sites)
    cfg.substeps = paper_cfg["substeps"]
    cfg.substeps_frame_dt = float(frame_dt)
    cfg.iterations = paper_cfg["iterations"]
    cfg.paper_self_contact = paper_cfg["particle_self_contact"]
    if paper_cfg.get("kinematic_mu") is not None:
        cfg.hand_cfg.kinematic_mu = paper_cfg["kinematic_mu"]
    known = PaperStackVBDCfg.__dataclass_fields__
    cfg.paper_cfg_overrides = {
        k: _tuples(v) for k, v in paper_cfg.items()
        if k in known and k not in _HAND_DERIVED_PAPER_FIELDS and k != "substeps"
    }
    cfg.paper_cfg_overrides["sites"] = sites
    return cfg


RETIRED_PAPER_CFG_FIELDS = {
    "kinematic_boxes": "box hand mirrors; a file with any is refused",
    "mirror_boxes": "box hand proxies; a file with any is refused",
    "sweep_kinematic_bodies": "the kinematic hand's substep sweep; a proxy file never read it",
    "use_cuda_graph": "the paper world's own graph; a proxy file stepped inside the solver's",
    "stack_center_xy": "the single-pile fallback; every file has sites",
    "stack_base_z": "the single-pile fallback; every file has sites",
    "num_sheets": "the single-pile fallback; every file has sites",
    "sheet_size_xy": "the single-pile fallback; every file has sites",
    "yaw_rad": "the single-pile fallback; every file has sites",
}


def recorded_cfg_diffs(recorded, rebuilt, path: str = "", tol: float = 1e-5) -> list[str]:
    """Where a rebuilt sidecar's JSON (``rebuilt``) differs from a recording's (``recorded``): the replay's
    and the augmentation's check that they run the paper and the carton the file was made with. A
    top-level ``RETIRED_PAPER_CFG_FIELDS`` key only the recording has is not a difference."""

    if isinstance(recorded, dict) and isinstance(rebuilt, dict):
        out = []
        for key in sorted(set(recorded) | set(rebuilt)):
            if not path and key in RETIRED_PAPER_CFG_FIELDS and key not in rebuilt:
                continue
            if key not in rebuilt or key not in recorded:
                out.append(f"{path}{key}: only in {'recording' if key in recorded else 'replay'}")
            else:
                out += recorded_cfg_diffs(recorded[key], rebuilt[key], f"{path}{key}.", tol)
        return out
    if isinstance(recorded, list) and isinstance(rebuilt, list):
        if len(recorded) != len(rebuilt):
            return [f"{path[:-1]}: length {len(recorded)} vs {len(rebuilt)}"]
        out = []
        for i, (a, b) in enumerate(zip(recorded, rebuilt)):
            out += recorded_cfg_diffs(a, b, f"{path}{i}.", tol)
        return out
    if isinstance(recorded, (int, float)) and isinstance(rebuilt, (int, float)) and not isinstance(recorded, bool):
        return [] if abs(recorded - rebuilt) <= tol else [f"{path[:-1]}: {recorded} vs {rebuilt}"]
    return [] if recorded == rebuilt else [f"{path[:-1]}: {recorded!r} vs {rebuilt!r}"]


def paper_substeps_per_frame(cfg: PackInvoiceSidecarCfg, frame_dt: float) -> int:
    """Paper substeps in one env frame of ``frame_dt``: ``cfg.substeps`` per ``cfg.substeps_frame_dt``."""

    return max(1, math.ceil(cfg.substeps * float(frame_dt) / cfg.substeps_frame_dt - 1e-9))


def set_active_pack_invoice_sidecar(sidecar: PackInvoiceSidecar | None) -> None:
    """Register the sidecar the last build produced (stack, coupler, carton)."""

    global _ACTIVE_SIDECAR
    _ACTIVE_SIDECAR = sidecar


def get_active_pack_invoice_sidecar() -> PackInvoiceSidecar | None:
    return _ACTIVE_SIDECAR

SITE_COLORS = ((0.93, 0.93, 0.90), (0.85, 0.90, 0.97))


@dataclass
class PaperHandCfg:
    """The robot hand as the paper world sees it: one proxy body per hand link, carrying that
    link's own collision hulls from the robot USD (the shapes the robot collides with)."""

    sides: tuple[str, ...] = ("left", "right")
    kinematic_mu: float = 1.2
    """Friction of the mirrored hand hulls, passed through to ``PaperStackVBDCfg.kinematic_mu``.
    The bench value (0.5) lets a pinched sheet slide out; a physical grasp needs the fingertips
    to bite harder than the surfaces the paper rests on."""
    hull_mirrors: dict[str, str] = field(default_factory=dict)
    """Mirror name -> robot link name, filled by the sidecar builder from the hand links that
    carry collision geometry."""


@dataclass
class PackInvoiceSidecarCfg:
    """Layout + solver knobs for the packing-bench paper sidecar [m]."""

    num_sheets: int = 4
    """Sheets per pile; the stacked pile the final grasp takes the top sheet off."""
    paper_self_contact: bool = True
    """Paper-paper contact (sheet on sheet, and a sheet folding onto itself).
    False: sheets exert no force on each other and only the bench, carton and
    hand push them, so they overlap in the pile. See
    ``PaperStackVBDCfg.particle_self_contact``."""
    substeps: int = 16
    """Paper substeps per ``substeps_frame_dt`` of simulated time; an env frame of another
    length gets the same paper time step (``paper_substeps_per_frame``). With proxy coupling
    the coupler splits them over the rigid solver steps (16 -> 4 per step at 30 Hz control,
    60 Hz physics, 2 MJWarp substeps: a 1/480 s paper step, the teleop recordings')."""
    substeps_frame_dt: float = 1.0 / 30.0
    """The frame ``substeps`` is stated for: the 30 Hz control step the teleop recorded at.
    The GA cells run a 60 Hz control step and get 8 substeps per frame, the same 1/480 s
    paper step, instead of 16 at half of it."""
    iterations: int = 10
    site_centers_xy: tuple[tuple[float, float], ...] = (E2_INVOICE_PILE_XY,)
    """Pile centres [m, world]: the invoice pile first; a second entry adds a flyer pile."""
    bench_slot: str = "Slot_PackingBench"
    """Prim (under the env) holding the bench the paper lies on."""
    bench_top: str = "BenchFC_Worktop"
    """The bench's worktop collider under ``bench_slot``: the paper's bench slab."""
    bench_obstacle_reach_m: float = 0.6
    """Also mirror the bench's colliders that stand on the worktop up to this height above
    it (monitor, roll, pegboard, posts, scale), as boxes, so a lifted sheet meets them."""
    carton_slot: str = "Slot_OutboundBox"
    """Prim holding the outbound box; its walls, floor and fill become the paper's carton."""
    station_entities: dict[str, str] = field(default_factory=lambda: {"bench": "packing_bench", "box": "outbound_box"})
    """The scene entities that carry the bench and the outbound box (``"bench"``, ``"box"`` -> entity
    name): at every reset the paper's copies of them are carried where those entities stand, the
    piles with the bench. An entity the scene lacks stays where the paper was built."""
    hand_coupling: bool = True
    """Mirror the robot's hand into the paper world as proxies inside the Newton MuJoCo solver's
    step (the teleop's ``one_way_proxy``). Off (the camera render, which writes the paper every
    frame and never steps it): no hand in the paper world, no coupler, and ``step`` refuses."""
    hand_cfg: PaperHandCfg = field(default_factory=PaperHandCfg)
    """The hand as the paper sees it: its own collision hulls."""
    sheet_texture: str | None = INVOICE_TEXTURE
    """Image (absolute path) shown on every sheet in place of the flat colour (a printed
    invoice by default); None draws the flat colour. Display only."""
    paper_cfg_overrides: dict[str, object] = field(default_factory=lambda: dict(TELEOP_PAPER_OVERRIDES))
    """Extra ``PaperStackVBDCfg`` fields (sheet_mass_kg, sheet_spacing_m,
    edge_ke, contact_mu, ...) applied on top of what this builder derives, so a
    headless experiment can sweep paper stock without a new builder per value."""
    coupling: ProxyCouplingCfg = field(default_factory=ProxyCouplingCfg)
    """Proxy pass knobs; see ``paper_coupling.py``."""


class PackInvoiceSidecar:
    """Bundle of the paper stack, the proxy coupler, and carton geometry."""

    def __init__(
        self,
        stack: PaperStackVBD,
        carton: OpenBoxCfg,
        *,
        coupled_solver=None,
        paper_pile=None,
    ):
        self.stack = stack
        self.paper_pile = paper_pile
        """The piles' per-reset draw (``PaperPileListener``, ``packing_bench.paper_pile``), or None."""
        self.coupled_solver = coupled_solver
        """The ``PaperProxyCoupler`` wrapping the main solver, or None (``hand_coupling`` off)."""
        self.carton = carton
        self.station_build_poses: dict[str, tuple[str, tuple, tuple]] = {}
        """``"bench"``/``"box"`` -> (scene entity, position, quat_xyzw) of that entity when the
        paper was built: what ``reset`` measures the entity's moves from."""

    def step(self, frame_dt: float) -> None:
        """Finish one env frame (call after ``env.step``): the paper advanced inside the main solver's step, and
        ``attach_pack_invoice_sidecar`` refreshes its meshes right before every render."""

        if self.coupled_solver is None:
            raise RuntimeError("this paper sidecar has no hand coupling (hand_coupling off) and is never stepped")

    def close(self) -> None:
        set_active_paper_stack(None)

    def follow_scene(self, env) -> None:
        """Carry the paper's bench and box where their scene entities now stand, and put the
        piles' next start on the moved bench at the paper-pile draw (``packing_bench.paper_pile``)."""

        env = getattr(env, "unwrapped", env)
        dz = 0.0
        for group, (entity, pos0, quat0) in self.station_build_poses.items():
            now = env.scene[entity].data.root_pose_w
            now = torch.as_tensor(now if isinstance(now, torch.Tensor) else wp.to_torch(now))[0].detach().cpu().double()
            q0 = torch.tensor(quat0, dtype=torch.float64)
            turn = quat_mul(now[3:7][None], quat_inv(q0[None]))
            shift = now[:3][None] - quat_apply(turn, torch.tensor(pos0, dtype=torch.float64)[None])
            self.stack.set_static_pose(group, shift[0].tolist(), turn[0].tolist())
            if group == "bench":
                dz = float(now[2] - pos0[2])
        dx, dy, dyaw = self.paper_pile.draw(0) if self.paper_pile is not None else (0.0, 0.0, 0.0)
        self.stack.reset_offset_xy = (dx, dy)
        self.stack.reset_yaw_rad = dyaw
        self.stack.reset_offset_z = dz
        self.carton = self.stack.box

    def reset(self, env=None) -> None:
        """Re-seed the piles and zero the coupler's (unused) feedback buffer (call on episode reset)."""

        if env is not None:
            self.follow_scene(env)
        if self.coupled_solver is not None:
            self.coupled_solver.reset_feedback()
        self.stack.reset()
        self.stack.update_usd_meshes()


def _world_bbox(stage, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    """World AABB of a prim, guide and proxy geometry included: colliders are often guide meshes."""

    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "guide", "proxy"])
    prim = stage.GetPrimAtPath(prim_path)
    assert prim.IsValid(), f"missing prim {prim_path}"
    bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return np.array(bound.GetMin()), np.array(bound.GetMax())


def _world_yaw(stage, prim_path: str) -> float:
    """A prim's heading about world +Z [rad], from its local-to-world transform."""

    from pxr import Usd, UsdGeom

    m = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    x_axis = m.TransformDir((1.0, 0.0, 0.0))
    return math.atan2(x_axis[1], x_axis[0])


def _yawed_bbox(stage, prim_path: str, yaw_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """``_world_bbox`` along axes turned ``yaw_rad`` about world +Z (a turned box's own axes)."""

    from pxr import Gf, Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "guide", "proxy"])
    prim = stage.GetPrimAtPath(prim_path)
    assert prim.IsValid(), f"missing prim {prim_path}"
    unturn = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), -math.degrees(yaw_rad)))
    bound = cache.ComputeWorldBound(prim)
    bound.Transform(unturn)
    aligned = bound.ComputeAlignedRange()
    return np.array(aligned.GetMin()), np.array(aligned.GetMax())


def _mesh_box(stage, mesh_path: str, frame_path: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The box a cuboid mesh (8 corners, any orientation) occupies, in the world or in
    ``frame_path``'s frame: centre, axes (rows, right-handed) and half extents."""

    from pxr import Usd, UsdGeom

    t = Usd.TimeCode.Default()
    prim = stage.GetPrimAtPath(mesh_path)
    local = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
    if len(local) != 8:
        raise ValueError(f"{mesh_path}: {len(local)} points, not a cuboid")
    to_frame = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(t)
    if frame_path is not None:
        to_frame = to_frame * UsdGeom.Xformable(stage.GetPrimAtPath(frame_path)).ComputeLocalToWorldTransform(t).GetInverse()
    xf = np.asarray(to_frame, dtype=float)
    pts = local @ xf[:3, :3] + xf[3, :3]
    center = pts.mean(0)
    axes = np.linalg.svd(pts - center)[2]
    if np.linalg.det(axes) < 0:
        axes[2] = -axes[2]
    proj = (pts - center) @ axes.T
    half = (proj.max(0) - proj.min(0)) / 2
    return center + axes.T @ ((proj.max(0) + proj.min(0)) / 2), axes, half


def _mesh_oriented_box(stage, mesh_path: str) -> StaticBoxCollider:
    """``_mesh_box`` in the world, as a static collider of the paper world."""

    center, axes, half = _mesh_box(stage, mesh_path)
    quat = quat_from_matrix(torch.as_tensor(axes.T, dtype=torch.float64)[None])[0]
    return StaticBoxCollider(
        center_xyz=tuple(float(v) for v in center),
        half_extents_xyz=tuple(float(v) for v in half),
        rotation_xyzw=tuple(float(v) for v in quat),
    )


_BOX_CORNER_SIGNS = np.array(
    [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], dtype=float
)
_BOX_TRIANGLES = np.array(
    [(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
     (3, 7, 6), (3, 6, 2), (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5)], dtype=np.int32
)


def _mesh_box_hull(stage, mesh_path: str, frame_path: str) -> tuple[np.ndarray, np.ndarray]:
    """``_mesh_box`` in ``frame_path``'s frame as one convex hull, (vertices, triangles):
    the form a ``kinematic_hulls`` mirror takes, which full-surface contact builds an
    SDF for (a primitive box on a proxy body gets no contacts: a sheet falls through it)."""

    center, axes, half = _mesh_box(stage, mesh_path, frame_path)
    return center + (_BOX_CORNER_SIGNS * half) @ axes, _BOX_TRIANGLES


def _find_prim_by_suffix(stage, root_path: str, suffix: str):
    from pxr import Usd

    root = stage.GetPrimAtPath(root_path)
    assert root.IsValid(), f"missing prim {root_path}"
    for prim in Usd.PrimRange(root):
        if prim.GetPath().pathString.endswith("/" + suffix):
            return prim
    raise AssertionError(f"no prim ending in /{suffix} under {root_path}")


def _resolve_main_bodies(rigid, env, mirror_links: dict[str, str]) -> dict[str, int]:
    """Mirror name -> main-solver body index, by the link name each mirror copies."""

    if env.num_envs != 1:
        raise ValueError(f"proxy paper coupling needs --num_envs 1, got {env.num_envs}")
    labels = [label.rsplit("/", 1)[-1] for label in rigid.model.body_label]
    mapping: dict[str, int] = {}
    for mirror, link in mirror_links.items():
        matches = [i for i, name in enumerate(labels) if name == link]
        if len(matches) != 1:
            raise ValueError(f"need exactly one main-solver body named {link}, found {len(matches)}")
        mapping[mirror] = matches[0]
    return mapping


def build_pack_invoice_sidecar(env, cfg: PackInvoiceSidecarCfg | None = None, hand: PaperHandSpec | None = None,
                               paper_args=None, paper_pile=None) -> PackInvoiceSidecar:
    """Build the pack-invoice paper sidecar against a reset environment."""

    import isaaclab.sim as sim_utils

    env = getattr(env, "unwrapped", env)
    refuse_retired_paper_env()
    cfg = cfg or get_pack_invoice_sidecar_cfg_override() or sidecar_cfg_from_args(paper_args)
    stage = sim_utils.get_current_stage()

    from pxr import Usd, UsdGeom, UsdPhysics

    bench_root = _find_prim_by_suffix(stage, ENV0, cfg.bench_slot).GetPath().pathString
    top_prim = _find_prim_by_suffix(stage, bench_root, cfg.bench_top)
    slab_min, slab_max = _world_bbox(stage, top_prim.GetPath().pathString)
    bench_top_z = float(slab_max[2])

    def colliders_under(root: str):
        from pxr import Usd

        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                yield prim

    sites: list[StackSiteCfg] = []
    for site_index, center_xy in enumerate(cfg.site_centers_xy):
        sites.append(
            StackSiteCfg(
                center_xy=(float(center_xy[0]), float(center_xy[1])),
                base_z=bench_top_z,
                num_sheets=cfg.num_sheets,
                sheet_size_xy=SHEET_SIZE_XY,
                color_rgb=SITE_COLORS[min(site_index, len(SITE_COLORS) - 1)],
                texture_path=cfg.sheet_texture if site_index == 0 else None,
            )
        )

    carton_flaps: dict[str, tuple[str, str | None]] = {}
    """Flap mesh name -> (its mesh, the body it moves with or None)."""
    box_root = _find_prim_by_suffix(stage, ENV0, cfg.carton_slot).GetPath().pathString
    found: dict[str, str] = {}
    for prim in colliders_under(box_root):
        name = prim.GetName()
        for key in ("Wall_E", "Wall_W", "Wall_N", "Wall_S", "Floor", "Fill"):
            if name.endswith(key) or f"_{key}_" in name:
                found[key] = prim.GetPath().pathString
    missing = {"Wall_E", "Wall_W", "Wall_N", "Wall_S", "Floor"} - set(found)
    if missing:
        raise ValueError(f"{box_root}: no collider for {sorted(missing)}; the outbound box changed shape")
    box_yaw = _world_yaw(stage, found["Floor"])
    parts = {key: _yawed_bbox(stage, path, box_yaw) for key, path in found.items()}
    x0, x1 = float(parts["Wall_W"][1][0]), float(parts["Wall_E"][0][0])
    y0, y1 = float(parts["Wall_S"][1][1]), float(parts["Wall_N"][0][1])
    rim_z = float(min(parts[k][1][2] for k in ("Wall_E", "Wall_W", "Wall_N", "Wall_S")))
    floor_z = float(max(parts["Floor"][1][2], parts["Fill"][1][2] if "Fill" in parts else -np.inf))
    wall_t = float(parts["Wall_W"][1][0] - parts["Wall_W"][0][0])
    center = rotate_points_by_yaw(torch.tensor([[(x0 + x1) / 2, (y0 + y1) / 2, 0.0]], dtype=torch.float64), box_yaw)
    carton = OpenBoxCfg(
        center_xy=(float(center[0, 0]), float(center[0, 1])), floor_z=floor_z,
        interior_half_x=(x1 - x0) / 2, interior_half_y=(y1 - y0) / 2,
        interior_depth=rim_z - floor_z, wall_thickness=wall_t, yaw_rad=box_yaw,
    )
    carton_from = box_root
    for prim in colliders_under(box_root):
        if "_Flap_" in prim.GetName():
            body = prim.GetParent()
            body_path = body.GetPath().pathString if body.HasAPI(UsdPhysics.RigidBodyAPI) else None
            carton_flaps[prim.GetName()] = (prim.GetPath().pathString, body_path)

    slab_c = (slab_min + slab_max) / 2
    extra_colliders = [StaticBoxCollider(
        center_xyz=tuple(float(v) for v in slab_c),
        half_extents_xyz=tuple(float(v) for v in (slab_max - slab_min) / 2),
    )]
    obstacles = []
    for prim in colliders_under(bench_root):
        if prim.GetPath() == top_prim.GetPath() or prim.GetPath().HasPrefix(top_prim.GetPath()):
            continue
        lo, hi = _world_bbox(stage, prim.GetPath().pathString)
        if hi[2] <= bench_top_z + 0.002 or lo[2] >= bench_top_z + cfg.bench_obstacle_reach_m:
            continue
        lo[2] = max(lo[2], bench_top_z)
        extra_colliders.append(StaticBoxCollider(
            center_xyz=tuple(float(v) for v in (lo + hi) / 2),
            half_extents_xyz=tuple(float(v) for v in (hi - lo) / 2),
        ))
        obstacles.append(prim.GetName())
    print(
        f"[pack_invoice] bench {top_prim.GetPath().pathString}: x[{slab_min[0]:.4f},{slab_max[0]:.4f}] "
        f"y[{slab_min[1]:.4f},{slab_max[1]:.4f}] top z {bench_top_z:.4f}; {len(obstacles)} obstacles on it; "
        f"carton from {carton_from}: interior {2 * carton.interior_half_x:.3f} x {2 * carton.interior_half_y:.3f} m, "
        f"floor z {carton.floor_z:.4f}, rim z {carton.rim_z:.4f}, {len(carton_flaps)} flaps ({sum(body is not None for _, body in carton_flaps.values())} hinged)"
    )

    robot = env.scene["robot"] if "robot" in env.scene.keys() else None
    proxy = cfg.hand_coupling and robot is not None
    rigid = main_mujoco_solver() if proxy else None
    if proxy and rigid is None:
        raise ValueError("the paper's proxy hand needs the Newton MuJoCo solver (--presets newton)")
    kinematic_hulls: dict[str, list] = {}
    mirror_links: dict[str, str] = {}
    if proxy:
        if hand is None:
            raise ValueError("the robot's embodiment declares no paper_hand, so the paper cannot mirror its hand")
        links = {
            f"hand_{side}_{link[len(side) + 1:]}": link
            for side in cfg.hand_cfg.sides
            for link in hand_hull_links(hand, list(robot.body_names), side)
        }
        hulls = link_collision_hulls(stage, list(links.values()))
        cfg.hand_cfg.hull_mirrors = {m: link for m, link in links.items() if link in hulls}
        kinematic_hulls = {m: hulls[link] for m, link in cfg.hand_cfg.hull_mirrors.items()}
        missing = sorted(set(links.values()) - set(hulls))
        print(f"[pack_invoice] paper sees the hand's own collision hulls: {len(kinematic_hulls)} links"
              + (f" (no collision mesh on {missing})" if missing else ""))
        mirror_links = dict(cfg.hand_cfg.hull_mirrors)
    hinged_flaps = {name: body for name, (_, body) in carton_flaps.items() if body is not None}
    if proxy and hinged_flaps:
        for name, body in hinged_flaps.items():
            kinematic_hulls[name] = [_mesh_box_hull(stage, carton_flaps[name][0], body)]
            mirror_links[name] = body.rsplit("/", 1)[-1]
    body_mapping: dict[str, int] | None = None
    if proxy:
        body_mapping = _resolve_main_bodies(rigid, env, mirror_links)

    paper_cfg_kwargs: dict[str, object] = dict(
        sites=tuple(sites),
        substeps=paper_substeps_per_frame(cfg, env.step_dt),
        iterations=cfg.iterations,
        particle_self_contact=cfg.paper_self_contact,
        kinematic_hulls=kinematic_hulls,
        kinematic_mu=cfg.hand_cfg.kinematic_mu if kinematic_hulls else None,
    )
    if proxy:
        paper_cfg_kwargs.update(
            proxy_mirrors=True, enable_rigid_soft_full_surface_contact=True,
        )
    static_flaps = [mesh for name, (mesh, _) in carton_flaps.items() if name not in kinematic_hulls]
    if hinged_flaps and static_flaps:
        print(f"[pack_invoice] box flaps are static in the paper world (no proxy coupler): {len(static_flaps)}")
    extra_colliders.extend(_mesh_oriented_box(stage, mesh) for mesh in static_flaps)
    if "substeps" in cfg.paper_cfg_overrides:
        raise ValueError("paper_cfg_overrides cannot set substeps; set PackInvoiceSidecarCfg.substeps")
    unknown = set(cfg.paper_cfg_overrides) - set(PaperStackVBDCfg.__dataclass_fields__)
    if unknown:
        raise ValueError(f"paper_cfg_overrides has no such PaperStackVBDCfg field(s): {sorted(unknown)}")
    paper_cfg_kwargs.update(cfg.paper_cfg_overrides)
    stack = PaperStackVBD(
        PaperStackVBDCfg(**paper_cfg_kwargs),
        box=carton,
        extra_colliders=tuple(extra_colliders),
    )
    set_active_paper_stack(stack)
    stack.create_usd_meshes(stage)

    coupled_solver = None
    if proxy:
        coupled_solver = install_paper_coupling(env, stack, body_mapping, cfg.coupling)
    else:
        capture_without_coupler()

    print(
        f"[pack_invoice] sidecar ready: piles {[tuple(round(v, 3) for v in s.center_xy) for s in sites]}, "
        f"carton {tuple(round(v, 3) for v in carton.center_xy)}, bench top {bench_top_z:.3f}, "
        f"hand coupling {'proxy' if coupled_solver is not None else 'off'}"
    )
    sidecar = PackInvoiceSidecar(
        stack, carton, coupled_solver=coupled_solver, paper_pile=paper_pile,
    )
    for group, entity in cfg.station_entities.items():
        if entity in env.scene.keys():
            prim = sim_utils.find_first_matching_prim(env.scene[entity].cfg.prim_path)
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            q = m.ExtractRotationQuat()
            sidecar.station_build_poses[group] = (
                entity, tuple(float(v) for v in m.ExtractTranslation()), (*(float(v) for v in q.GetImaginary()), float(q.GetReal())),
            )
    set_active_pack_invoice_sidecar(sidecar)
    return sidecar
