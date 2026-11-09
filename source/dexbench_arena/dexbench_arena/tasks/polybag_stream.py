# SPDX-License-Identifier: Apache-2.0
"""DB-L1's item stream: polybags and boxes put down on the moving belt one after another, and each item judged."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from isaaclab.envs.mdp import time_out
from isaaclab.managers import ManagerTermBase, RecorderManager
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz
from isaaclab_arena.utils.yaw import rotate_points_by_yaw_batch

from dexbench_arena.assets.polybag.sort_variants import BOX_VARIANTS_BY_KEY, SORT_VARIANTS_BY_KEY
from dexbench_arena.evaluation.replay_guards import ReplayGuardError, draws_through_variation
from dexbench_arena.tasks.polybag_stream_plan import PlanEpisode, alternating_kinds, off_belt_deg, split_layout_draw
from dexbench_arena.tasks.polybag_stream_record import EPISODE_RECORD_VERSION, STATE_NAMES, placed_in_order_steps

WAITING, IN_PLAY, PLACED, WRONG_TOTE, DROPPED, MISSED = range(6)
FAILED_STATES = (WRONG_TOTE, DROPPED, MISSED)
DRAWN, SCHEDULED, RESUMED, LAID = -1, -2, -3, -4
"""``PolybagStream.plan_entry`` of an episode drawn at random, one ``schedule_next`` set, a resumed segment, and one laid
out from the environment's layout draw (``set_layout_draw``)."""
MODES = ("episode", "continuous")
LAYOUT_VARIATION_EVENT = "l1_stream_layout_variation"
"""The event Arena composes for DB-L1's layout variation (``L1StreamLayoutVariation.build_event_cfg``): the one way an
episode's layout comes from a draw the replay guard accepts (``replay_guards.draws_through_variation``)."""
_BAG, _BOX = 0, 1


@dataclass
class StreamItem:
    """One item of the stream. Not frozen: Isaac Lab's configclass walks an event term's params and writes back
    into the dataclasses it finds there."""

    kind: str
    """"bag" or "box"."""
    entity: str
    """Scene entity name."""
    half_extent_xy: tuple[float, float]
    """Half the item's footprint along x and y at its rest pose [m] (a bag's nominal footprint, a box's size)."""
    park_pose: tuple[float, ...]
    """Where it waits: (x, y, z, qx, qy, qz, qw) of its asset frame, environment frame (a box's origin is on its
    bottom face)."""
    target_tote: int = -1
    """A bag's target tote (index into ``tote_boxes``; 0 with one tote); -1 for a box."""
    height: float = 0.0
    """A box's height [m]."""
    variant: str = ""
    """Its pool variant (``sort_variants.SORT_VARIANTS_BY_KEY`` or ``BOX_VARIANTS_BY_KEY``). Read by
    ``episode_record`` only."""


def long_axis_angle(xy: torch.Tensor) -> torch.Tensor:
    """The direction of the principal (long) axis of point sets ``xy`` [..., P, 2], as an angle about z [rad]."""
    d = xy - xy.mean(dim=-2, keepdim=True)
    sxx, syy, sxy = (d[..., 0] ** 2).mean(-1), (d[..., 1] ** 2).mean(-1), (d[..., 0] * d[..., 1]).mean(-1)
    return 0.5 * torch.atan2(2.0 * sxy, sxx - syy)


def _active_recorder(env):
    """The env's recorder manager when it records (record_demos.py); None in teleop.py, an evaluation, a stand-in."""
    manager = getattr(env, "recorder_manager", None)
    return manager if manager is not None and len(manager.active_terms) > 0 else None


@draws_through_variation(LAYOUT_VARIATION_EVENT)
class PolybagStream(ManagerTermBase):
    """The DB-L1 item stream and its judging (see the module docstring). The task's event term ``l1_stream``
    (``STREAM_TERM``); the termination terms and tools read its state through :func:`l1_stream`."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        self.items: tuple[StreamItem, ...] = tuple(p["items"])
        self.bags = [i for i, it in enumerate(self.items) if it.kind == "bag"]
        self.boxes = [i for i, it in enumerate(self.items) if it.kind == "box"]
        if not self.bags:
            raise ValueError("PolybagStream: no bag in the stream")
        if float(p["belt_speed"]) <= 0.0:
            raise ValueError(f"PolybagStream: the stream needs a moving belt, not {p['belt_speed']} m/s")
        self.mode = p.get("mode", "episode")
        if self.mode not in MODES:
            raise ValueError(f"PolybagStream: mode {self.mode!r}, one of {MODES}")
        self.episode_bags = int(p.get("episode_bags") or len(self.bags))
        self.episode_boxes = len(self.boxes) if p.get("episode_boxes") is None else int(p["episode_boxes"])
        if not 1 <= self.episode_bags <= len(self.bags):
            raise ValueError(f"PolybagStream: episode_bags {self.episode_bags}, 1 to the scene's {len(self.bags)}")
        if not 0 <= self.episode_boxes <= len(self.boxes):
            raise ValueError(f"PolybagStream: episode_boxes {self.episode_boxes}, 0 to the scene's {len(self.boxes)}")
        self.kinds = alternating_kinds(p["lead"], self.episode_bags, self.episode_boxes)
        self.headline_bags = int(p.get("headline_bags") or 0)
        """Episode mode: the success is the first that many bags placed in stream order before any failure; 0: every
        bag placed and no box disturbed (``succeeded``), as a collection's and a replay's episodes are judged."""
        if self.headline_bags and (self.mode != "episode" or not 1 <= self.headline_bags <= self.episode_bags):
            raise ValueError(f"PolybagStream: headline_bags {self.headline_bags} takes an episode stream and 1 to its "
                             f"{self.episode_bags} bags (mode {self.mode})")
        n, e, dev = len(self.items), env.num_envs, env.device
        self.release_t = torch.zeros(e, n, device=dev)
        """Episode mode: when each item goes down [s from the episode's start]; inf for one left out of it."""
        self.used = torch.zeros(e, n, dtype=torch.bool, device=dev)
        """The items of the env's episode (episode mode; every item in continuous mode)."""
        self.lateral_y = torch.zeros(e, n, device=dev)
        """Where across the belt each item's centre goes down [m, env frame]; once it is down, where it went."""
        self.yaw = torch.zeros(e, n, device=dev)
        """Each item's turn about z from its rest pose [rad]."""
        self.released = torch.zeros(e, n, dtype=torch.bool, device=dev)
        self.box_live = torch.zeros(e, n, dtype=torch.bool, device=dev)
        """A released box not parked yet: on the belt, in a hand, or wherever it was knocked."""
        self.state = torch.full((e, n), WAITING, dtype=torch.long, device=dev)
        self.rest_s = torch.zeros(e, n, device=dev)
        self.rest_tote = torch.full((e, n), -1, dtype=torch.long, device=dev)
        self.inside = torch.zeros(e, n, device=dev)
        self.elapsed = torch.zeros(e, device=dev)
        self.episode = torch.zeros(e, dtype=torch.long, device=dev)
        self.planned_release_t = torch.zeros(e, n, device=dev)
        """The release times the episode started with (``release_t`` is pushed back when the belt is blocked)."""
        self.release_step = torch.full((e, n), -1, dtype=torch.long, device=dev)
        """The stream step (``env.episode_length_buf`` + ``step_offset``) the item was last released in; -1: not yet."""
        self.final_step = torch.full((e, n), -1, dtype=torch.long, device=dev)
        """The stream step a bag got its final state in, or a released box was parked in; -1: not yet."""
        self.step_offset = torch.zeros(e, dtype=torch.long, device=dev)
        """Added to ``env.episode_length_buf`` for the stream's step: a resumed segment's start step, else 0."""
        self.pick_zone_off_belt = torch.full((e, n), math.nan, device=dev)
        """A bag's long side's angle off the belt's direction when its centre first reached ``pick_zone_x`` lying on
        the belt [rad, 0..pi/2], from its particles (it can turn on the belt, against a cover or a box); nan: not
        yet (or picked up before)."""
        self.ref_y = torch.zeros(e, n, device=dev)
        self.ref_yaw = torch.zeros(e, n, device=dev)
        """A released box's centre across the belt [m] and heading [rad] as it went down: where the belt carries it."""
        self.disturbed = torch.zeros(e, n, dtype=torch.bool, device=dev)
        self.disturbed_step = torch.full((e, n), -1, dtype=torch.long, device=dev)
        self.deviation = torch.zeros(e, n, 3, device=dev)
        """A box's largest offset from the belt's carry since it went down: across [m], heading [rad], lift [m]."""
        self.tote_lo = torch.tensor([b[0] for b in p["tote_boxes"]], device=dev)
        self.tote_hi = torch.tensor([b[1] for b in p["tote_boxes"]], device=dev)
        self._rest: dict[int, torch.Tensor] = {}
        self.rest_long_axis = torch.zeros(n, device=dev)
        """Each item's long side's direction at its rest pose [rad, about z from +X]."""
        for k, it in enumerate(self.items):
            if it.kind == "bag":
                rest = torch.as_tensor(env.scene[it.entity].template.rest, dtype=torch.float32, device=dev)
                self._rest[k] = rest - rest.mean(dim=0, keepdim=True)
                self.rest_long_axis[k] = long_axis_angle(self._rest[k][None, :, :2])[0]
            else:
                hx, hy = it.half_extent_xy
                self.rest_long_axis[k] = 0.0 if hx >= hy else math.pi / 2.0
        self._scheduled: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._layout_draws: dict[int, list[float]] = {}
        self._resume: dict[int, dict] = {}
        self.last_summary: dict[int, dict[str, int]] = {}
        """Per env, how its last episode's bags ended (``summary``), kept at its reset."""
        self.placed_total = torch.zeros(e, n, dtype=torch.long, device=dev)
        self.released_total = torch.zeros(e, n, dtype=torch.long, device=dev)
        self.tote_count = torch.zeros(e, dtype=torch.long, device=dev)
        self.next_kind = torch.zeros(e, dtype=torch.long, device=dev)
        self.next_t = torch.zeros(e, device=dev)
        self._next_draw: list[tuple[float, float, bool, bool]] = [(0.0, 0.0, False, False)] * e
        self._rng = [torch.Generator(device="cpu") for _ in range(e)]
        self.segment = torch.zeros(e, dtype=torch.long, device=dev)
        """The running segment's index in the stream (continuous mode)."""
        self.segments_kept = torch.zeros(e, dtype=torch.long, device=dev)
        self.segments_discarded = torch.zeros(e, dtype=torch.long, device=dev)
        self._segment_start: list[dict] = [{} for _ in range(e)]
        self._segment_events: list[list[dict]] = [[] for _ in range(e)]
        self._forced: list[list[tuple[int, int, float, float]]] = [[] for _ in range(e)]
        """A resumed segment's releases: (stream step, item, lateral y, turn), in step order."""
        self.replaying = [False] * e
        """The env runs a resumed segment (a replay tool's): the stream never drives the recorder there."""
        self.resumed_placed = torch.zeros(e, dtype=torch.bool, device=dev)
        """A resumed segment's bag was placed again (continuous mode): the replay's success (``succeeded``)."""
        self._closing: dict[int, dict] = {}
        self.plan: tuple[PlanEpisode, ...] = tuple(p.get("plan") or ())
        start = int(p.get("plan_start") or 0)
        if self.plan and self.mode != "episode":
            raise ValueError("PolybagStream: a collection plan lays episodes out; a continuous stream has none")
        if self.plan and e != 1:
            raise ValueError(f"PolybagStream: a collection plan runs one environment, not {e}")
        self.require_layout_draw = bool(p.get("require_layout_draw", False))
        """A benchmark replay: every episode is laid out from the layout draw, never the stream's own (``reset``)."""
        if self.require_layout_draw and (self.continuous or self.plan):
            what = "draws each item as it goes (continuous)" if self.continuous else "runs a collection plan"
            raise ValueError("PolybagStream: a benchmark replay lays every episode out from its rows' layout draws; "
                             f"this stream {what}")
        if self.plan and not 0 <= start < len(self.plan):
            raise ValueError(f"PolybagStream: plan_start {start}: the plan has entries 0-{len(self.plan) - 1}")
        self._plan_cursor = [start] * e
        self.plan_entry = torch.full((e,), DRAWN, dtype=torch.long, device=dev)
        """The plan entry the episode runs (``DRAWN``, ``SCHEDULED``, ``RESUMED`` or an index into ``plan``)."""
        self.plan_attempt = torch.zeros(e, dtype=torch.long, device=dev)
        """How many episodes have started on that entry, this one included (0 off the plan)."""

    @property
    def continuous(self) -> bool:
        return self.mode == "continuous"


    def _origin(self, env_ids):
        return self._env.scene.env_origins[env_ids]

    def _turned_rest(self, k: int, yaw: torch.Tensor) -> torch.Tensor:
        """(len(yaw), P, 3) bag ``k``'s rest shape about its mean turned by ``yaw``, its lowest point at z 0."""
        rest = self._rest[k]
        turned = rotate_points_by_yaw_batch(rest.repeat(len(yaw), 1), yaw.repeat_interleave(len(rest)))
        turned = turned.reshape(len(yaw), len(rest), 3)
        turned[..., 2] -= rest[:, 2].min()
        return turned

    def _extent_xy(self, k: int, yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The xy extent of item ``k`` turned by ``yaw`` (per env), as (min, max) offsets from where it is placed: a
        bag's rest shape about its particles' mean (what ``_place_bag`` centres), a box's rectangle."""
        it = self.items[k]
        if it.kind == "bag":
            xy = self._turned_rest(k, yaw)[..., :2]
            return xy.amin(dim=1), xy.amax(dim=1)
        hx, hy = it.half_extent_xy
        corners = torch.tensor([[hx, hy, 0.0], [hx, -hy, 0.0]], device=yaw.device).repeat(len(yaw), 1)
        turned = rotate_points_by_yaw_batch(corners, yaw.repeat_interleave(2)).reshape(len(yaw), 2, 3)
        half = turned[..., :2].abs().amax(dim=1)
        return -half, half

    def _box_pose(self, k: int, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Box ``k``'s position (its bottom face's centre, env frame) and its yaw, per env."""
        pose = self._env.scene[self.items[k].entity].data.root_link_pose_w.torch[env_ids]
        _, _, yaw = euler_xyz_from_quat(pose[:, 3:7])
        return pose[:, :3] - self._origin(env_ids), yaw

    def _on_belt_boxes_xy(self, e: int) -> list[tuple[float, float, float, float]]:
        """(xmin, ymin, xmax, ymax) of what is out in env ``e``: bags in play (their particles), boxes not parked."""
        out = []
        ids = torch.tensor([e], device=self._env.device)
        for k, it in enumerate(self.items):
            if not bool(self.released[e, k]):
                continue
            if it.kind == "bag":
                if int(self.state[e, k]) != IN_PLAY:
                    continue
                xy = self._env.scene[it.entity].data.particle_pos_w.torch[e, :, :2] - self._origin(ids)[0, :2]
                lo, hi = xy.amin(dim=0), xy.amax(dim=0)
            else:
                if not bool(self.box_live[e, k]):
                    continue
                pos, yaw = self._box_pose(k, ids)
                lo_off, hi_off = self._extent_xy(k, yaw)
                lo, hi = pos[0, :2] + lo_off[0], pos[0, :2] + hi_off[0]
            out.append((float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])))
        return out

    def _place_bag(self, k: int, env_ids: torch.Tensor, centre_xy: torch.Tensor, base_z: float, yaw: torch.Tensor,
                   vel_x: float) -> None:
        """Put bag ``k``'s particles, in their rest shape turned by ``yaw``, with their mean at ``centre_xy`` and their
        lowest point at ``base_z`` (env frame), moving at ``vel_x`` along x, and make it active."""
        bag = self._env.scene[self.items[k].entity]
        origin = self._origin(env_ids)
        pos = self._turned_rest(k, yaw) + origin[:, None, :]
        pos[..., 0] += centre_xy[:, None, 0]
        pos[..., 1] += centre_xy[:, None, 1]
        pos[..., 2] += base_z
        vel = torch.zeros_like(pos)
        vel[..., 0] = vel_x
        bag.write_nodal_state_to_sim_index(torch.cat([pos, vel], dim=-1), env_ids=env_ids)
        bag.write_active_to_sim_index(True, env_ids)

    def _place_box(self, k: int, env_ids: torch.Tensor, xy: torch.Tensor, z: float, yaw: torch.Tensor,
                   vel_x: float = 0.0) -> None:
        """Put box ``k`` at ``xy``, its bottom at ``z`` (env frame), turned by ``yaw``, moving at ``vel_x`` along x."""
        box = self._env.scene[self.items[k].entity]
        origin = self._origin(env_ids)
        pose = torch.zeros(len(env_ids), 7, device=self._env.device)
        pose[:, 0] = xy[:, 0] + origin[:, 0]
        pose[:, 1] = xy[:, 1] + origin[:, 1]
        pose[:, 2] = z + origin[:, 2]
        pose[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        box.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        vel = torch.zeros(len(env_ids), 6, device=self._env.device)
        vel[:, 0] = vel_x
        box.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)

    def _park(self, k: int, env_ids: torch.Tensor) -> None:
        """Item ``k`` back in its slot on the park slab, at rest; a bag out of the simulation."""
        it, dev = self.items[k], self._env.device
        park = torch.tensor(it.park_pose, dtype=torch.float32, device=dev)[None].repeat(len(env_ids), 1)
        park[:, :3] += self._origin(env_ids)
        asset = self._env.scene[it.entity]
        if it.kind == "bag":
            asset.write_rest_pose_to_sim_index(park[:, :3], park[:, 3:], env_ids)
            asset.write_active_to_sim_index(False, env_ids)
        else:
            asset.write_root_pose_to_sim_index(root_pose=park, env_ids=env_ids)
            asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(env_ids), 6, device=dev),
                                                   env_ids=env_ids)
            self.box_live[env_ids, k] = False

    def _free(self, e: int, k: int) -> None:
        """Continuous mode: item ``k`` of env ``e``, parked, waits to go down again."""
        self.released[e, k] = False
        self.state[e, k] = WAITING
        self.release_step[e, k] = -1
        self.final_step[e, k] = -1
        self.disturbed[e, k] = False
        self.disturbed_step[e, k] = -1
        self.deviation[e, k] = 0.0


    def schedule_next(self, env_id: int, release_t, lateral_y, yaw_rad) -> None:
        """Release env ``env_id``'s next episode on this schedule instead of a drawn one: per item (``self.items``
        order) its release time [s from the episode's start; ``math.inf`` leaves the item out of the episode], lateral
        y [m, env frame] and turn [rad]. For tools that generate episodes and replays (episode mode); a blocked item
        still waits for the belt to clear."""
        n, dev = len(self.items), self._env.device
        values = [torch.as_tensor(v, dtype=torch.float32, device=dev).reshape(-1) for v in (release_t, lateral_y, yaw_rad)]
        if any(v.numel() != n for v in values):
            raise ValueError(f"schedule_next: {n} items, got {[v.numel() for v in values]} values")
        if self.continuous:
            raise ValueError("schedule_next lays an episode out; a continuous stream is resumed (schedule_resume)")
        times, rest = values[0], torch.stack(values[1:])
        if bool(torch.isnan(times).any()) or bool((times < 0).any()) or not bool(torch.isfinite(rest).all()):
            raise ValueError("schedule_next: release times must be non-negative (inf: not in the episode), lateral "
                             "positions and turns finite")
        if not bool(torch.isfinite(times[self.bags]).any()):
            raise ValueError("schedule_next: no bag of the episode is released")
        self._scheduled[int(env_id)] = tuple(values)

    def set_layout_draw(self, env_id: int, draw) -> None:
        """Lay env ``env_id``'s next episode out from ``draw`` (episode mode), the values of an episode's layout draw
        (``polybag_stream_plan.split_layout_draw``: an ordering key per bag and box of the scene, then five values per
        release) instead of the stream's own draw. A ``schedule_next`` layout or a plan entry comes first; the draw
        is used at the env's next reset and dropped there either way. ValueError for a draw of another size or
        outside [0, 1], or a continuous stream."""
        if self.continuous:
            raise ValueError("set_layout_draw lays an episode out; a continuous stream draws each item as it goes")
        values = [float(v) for v in torch.as_tensor(draw).reshape(-1).tolist()]
        split_layout_draw(values, len(self.bags), len(self.boxes), len(self.kinds))
        self._layout_draws[int(env_id)] = values

    def schedule_resume(self, env_id: int, segment: dict) -> None:
        """Run env ``env_id``'s next reset as a recorded segment of a continuous stream (a demo's
        ``episode_record()["segment"]``): the stream's state at the segment's start (which items are out, where they
        went down, the judge's timers, the tote's load) and its releases at the recorded steps, without drawing or
        waiting for the belt. The scene's own state is the demo's ``initial_state``, which the replay's ``reset_to``
        writes after this reset. ValueError for a segment of another stream (other items)."""
        if not self.continuous:
            raise ValueError("schedule_resume resumes a continuous stream; this one runs episodes (l1_stream_mode)")
        start = segment["start"]
        releases = [event for event in segment["events"] if event["what"] == "release"]
        items = [(item["entity"], item["variant"]) for item in start["items"]]
        if items != [(it.entity, it.variant or None) for it in self.items]:
            raise ValueError(f"schedule_resume: the segment's items {items} are not this stream's "
                             f"{[(it.entity, it.variant or None) for it in self.items]}")
        self._resume[int(env_id)] = segment
        self._forced[int(env_id)] = [(int(r["step"]), int(r["item"]), float(r["lateral_y_m"]), float(r["yaw_rad"]))
                                     for r in releases]

    def reset(self, env_ids=None) -> None:
        dev = self._env.device
        ids = torch.arange(self._env.num_envs, device=dev) if env_ids is None else torch.as_tensor(env_ids, device=dev)
        if len(ids) == 0:
            return
        kept = self.succeeded()
        for e in ids.tolist():
            ended = bool(self.released[e].any())
            if ended and not self.continuous:
                self.last_summary[e] = self.summary(e)
                tally = ", ".join(f"{name} {count}" for name, count in self.last_summary[e].items() if count)
                print(f"[dexbench] L1 stream episode {int(self.episode[e])} (env {e}): {tally} of {self.episode_bags} "
                      f"bags", flush=True)
            if self.continuous and (ended or int(self.segment[e])):
                print(f"[dexbench] L1 stream {int(self.episode[e])} (env {e}) ends: {int(self.segments_kept[e])} bags "
                      f"placed and kept, {int(self.segments_discarded[e])} segments discarded", flush=True)
            previous = int(self.plan_entry[e])
            if ended and previous >= 0 and bool(kept[e]):
                self._plan_cursor[e] = previous + 1
            self.episode[e] += 1
            self.step_offset[e] = 0
            self.replaying[e] = False
            self.resumed_placed[e] = False
            if self.continuous:
                self.release_t[e] = math.inf
                self.lateral_y[e] = 0.0
                self.yaw[e] = 0.0
                self.plan_entry[e], self.plan_attempt[e] = (RESUMED if e in self._resume else DRAWN), 0
                continue
            scheduled = self._scheduled.pop(e, None)
            laid = self._layout_draws.pop(e, None)
            if scheduled is not None:
                self.release_t[e], self.lateral_y[e], self.yaw[e] = scheduled
                self.plan_entry[e], self.plan_attempt[e] = SCHEDULED, 0
                print(f"[dexbench] L1 stream episode {int(self.episode[e])} (env {e}), scheduled: "
                      f"{self._layout_text(e)}", flush=True)
            elif self._plan_cursor[e] < len(self.plan):
                cursor = self._plan_cursor[e]
                entry = self.plan[cursor]
                self.release_t[e] = torch.tensor(entry.release_t, dtype=torch.float32, device=dev)
                self.lateral_y[e] = torch.tensor(entry.lateral_y, dtype=torch.float32, device=dev)
                self.yaw[e] = torch.tensor(entry.yaw_rad, dtype=torch.float32, device=dev)
                if previous != cursor:
                    self.plan_attempt[e] = 1
                elif ended:
                    self.plan_attempt[e] += 1
                self.plan_entry[e] = cursor
                print(f"[dexbench] L1 plan episode {cursor + 1}/{len(self.plan)} (id {entry.id}, try "
                      f"{int(self.plan_attempt[e])}, env {e}): {self._layout_text(e)}", flush=True)
            elif laid is not None:
                bag_order, box_order, places = split_layout_draw(laid, len(self.bags), len(self.boxes), len(self.kinds))
                columns = [torch.tensor(column, dtype=torch.float32) for column in zip(*places)]
                self._lay_out(e, bag_order, box_order, *columns[:3], columns[3] < 0.5, columns[4] < 0.5)
                self.plan_entry[e], self.plan_attempt[e] = LAID, 0
                print(f"[dexbench] L1 stream episode {int(self.episode[e])} (env {e}), from the layout draw: "
                      f"{self._layout_text(e)}", flush=True)
            elif self.require_layout_draw:
                raise ReplayGuardError(
                    f"[DexBench benchmark replay] L1 stream episode {int(self.episode[e])} (env {e}) has no layout draw: "
                    f"the {LAYOUT_VARIATION_EVENT} event did not hand it one at this reset, and a benchmark replay never "
                    "lays an episode out from the stream's own seed (no frozen row records that draw)")
            else:
                if self.plan:
                    print(f"[dexbench] L1 plan: all {len(self.plan)} entries done; episode {int(self.episode[e])} "
                          f"(env {e}) is drawn at random", flush=True)
                self._draw(e)
                self.plan_entry[e], self.plan_attempt[e] = DRAWN, 0
                print(f"[dexbench] L1 stream episode {int(self.episode[e])} (env {e}): {self._layout_text(e)}",
                      flush=True)
        self.used[ids] = True if self.continuous else torch.isfinite(self.release_t[ids])
        self.planned_release_t[ids] = self.release_t[ids]
        self.release_step[ids] = -1
        self.final_step[ids] = -1
        self.pick_zone_off_belt[ids] = math.nan
        self.released[ids] = False
        self.state[ids] = WAITING
        self.rest_s[ids] = 0.0
        self.rest_tote[ids] = -1
        self.inside[ids] = 0.0
        self.elapsed[ids] = 0.0
        self.disturbed[ids] = False
        self.disturbed_step[ids] = -1
        self.deviation[ids] = 0.0
        self.ref_y[ids] = 0.0
        self.ref_yaw[ids] = 0.0
        for k in range(len(self.items)):
            self._park(k, ids)
        if self.continuous:
            for e in ids.tolist():
                self._start_stream(e)

    def _draw(self, e: int) -> None:
        """Draw env ``e``'s episode (module docstring), seeded by its seed, episode and env: ``episode_bags`` of the
        bags and ``episode_boxes`` of the boxes, each in random order, alternating from the lead."""
        p = self.cfg.params
        g = torch.Generator(device="cpu").manual_seed(int(p["seed"]) * 100003 + (int(self.episode[e]) - 1) * 7919 + e)
        bag_order = torch.randperm(len(self.bags), generator=g).tolist()
        box_order = torch.randperm(len(self.boxes), generator=g).tolist()
        m = len(self.kinds)
        gap, lateral = torch.rand(m, generator=g), torch.rand(m, generator=g)
        unit, sign, flip = torch.rand(m, generator=g), torch.rand(m, generator=g) < 0.5, torch.rand(m, generator=g) < 0.5
        self._lay_out(e, bag_order, box_order, gap, lateral, unit, sign, flip)

    def _lay_out(self, e: int, bag_order: list[int], box_order: list[int], gap: torch.Tensor, lateral: torch.Tensor,
                 unit: torch.Tensor, sign: torch.Tensor, flip: torch.Tensor) -> None:
        """Lay env ``e``'s episode out: its bags and boxes taken in ``bag_order`` / ``box_order`` (indices into the
        scene's bags and boxes) alternating from the lead, and per place of the release order the belt's run from the
        previous release (``gap``, a fraction of the spacing range; the first goes down at once), the lateral centre
        (a fraction of the lateral range), the turn's fraction of its kind's band and its two coin flips."""
        p = self.cfg.params
        queues = {"bag": [self.bags[i] for i in bag_order], "box": [self.boxes[i] for i in box_order]}
        order = [queues[kind].pop(0) for kind in self.kinds]
        low, high = (float(v) for v in p["spacing_m"])
        gaps = low + (high - low) * gap
        gaps[0] = 0.0
        times = torch.cumsum(gaps, 0) / float(p["belt_speed"])
        lo, hi = (float(v) for v in p["lateral_y"])
        lateral = lo + (hi - lo) * lateral
        self.release_t[e] = math.inf
        self.lateral_y[e] = 0.0
        self.yaw[e] = 0.0
        for place, k in enumerate(order):
            self.yaw[e, k] = self._turn(k, float(unit[place]), bool(sign[place]), bool(flip[place]))
            self.release_t[e, k] = float(times[place])
            self.lateral_y[e, k] = float(lateral[place])

    def _turn(self, k: int, unit: float, sign: bool, flip: bool) -> float:
        """Item ``k``'s turn [rad] for a drawn fraction ``unit`` of its kind's band and the two coin flips."""
        p = self.cfg.params
        band = p["bag_yaw_off_belt_deg"] if self.items[k].kind == "bag" else p["box_yaw_off_belt_deg"]
        off = math.radians(float(band[0]) + (float(band[1]) - float(band[0])) * unit)
        long_axis = (off if sign else -off) + (math.pi if flip else 0.0)
        return math.remainder(long_axis - float(self.rest_long_axis[k]), 2.0 * math.pi)

    def _label(self, k: int) -> str:
        it = self.items[k]
        if it.kind == "box":
            return f"box{self.boxes.index(k)}"
        return f"bag{self.bags.index(k)} {it.variant or it.entity}"

    def _off_belt_deg(self, e: int, k: int) -> float:
        """Item ``k``'s long side's angle off the belt's direction as turned in env ``e`` [deg, 0-90]."""
        return off_belt_deg(math.degrees(float(self.rest_long_axis[k]) + float(self.yaw[e, k])))

    def _item_text(self, e: int, k: int) -> str:
        return f"{self._label(k)} {self._off_belt_deg(e, k):.0f} deg off the belt y {float(self.lateral_y[e, k]):.2f}"

    def _layout_text(self, e: int) -> str:
        """Env ``e``'s items of the episode in release order, each with its angle off the belt and lateral centre (the
        operator's log)."""
        ks = [k for k in range(len(self.items)) if math.isfinite(float(self.release_t[e, k]))]
        return ", ".join(self._item_text(e, k) for k in sorted(ks, key=lambda k: (float(self.release_t[e, k]), k)))

    def _put_down(self, env, e: int, k: int, lateral: float, yaw: float, step: torch.Tensor, *,
                  check_clear: bool = True) -> bool:
        """Put item ``k`` of env ``e`` down at the entry, its centre at ``lateral`` (moved onto the clear belt) and
        turned by ``yaw``, unless ``check_clear`` and its footprint does not clear the belt; whether it went down."""
        p, dev = self.cfg.params, env.device
        it = self.items[k]
        yaw_t = torch.tensor([yaw], dtype=torch.float32, device=dev)
        lo_off, hi_off = (v[0].tolist() for v in self._extent_xy(k, yaw_t))
        x = float(p["entry_x"]) - lo_off[0]
        margin = float(p["edge_margin"])
        y_min = float(p["clear_y"][0]) + margin - lo_off[1]
        y_max = float(p["clear_y"][1]) - margin - hi_off[1]
        y = (y_min + y_max) / 2.0 if y_min > y_max else min(max(float(lateral), y_min), y_max)
        if check_clear:
            clearance = float(p["spawn_clearance"])
            lo = (x + lo_off[0] - clearance, y + lo_off[1] - clearance)
            hi = (x + hi_off[0] + clearance, y + hi_off[1] + clearance)
            if any(b[0] < hi[0] and lo[0] < b[2] and b[1] < hi[1] and lo[1] < b[3] for b in self._on_belt_boxes_xy(e)):
                return False
        self.lateral_y[e, k] = y
        self.yaw[e, k] = yaw_t[0]
        e_ids = torch.tensor([e], device=dev)
        xy = torch.tensor([[x, y]], device=dev)
        speed, belt_top = float(p["belt_speed"]), float(p["belt_top_z"])
        if it.kind == "bag":
            self._place_bag(k, e_ids, xy, belt_top + float(p["bag_clearance"]), yaw_t, speed)
            self.state[e, k] = IN_PLAY
        else:
            self._place_box(k, e_ids, xy, belt_top + float(p["box_clearance"]), yaw_t, speed)
            self.box_live[e, k] = True
        self.released[e, k] = True
        self.release_step[e, k] = step[e]
        self.final_step[e, k] = -1
        self.rest_s[e, k] = 0.0
        self.rest_tote[e, k] = -1
        self.inside[e, k] = 0.0
        self.pick_zone_off_belt[e, k] = math.nan
        self.ref_y[e, k] = self.lateral_y[e, k]
        self.ref_yaw[e, k] = self.yaw[e, k]
        self.disturbed[e, k] = False
        self.disturbed_step[e, k] = -1
        self.deviation[e, k] = 0.0
        return True

    def _release(self, env, e: int, step: torch.Tensor) -> None:
        """Episode mode: put env ``e``'s earliest due item down, if its footprint at the entry clears the belt."""
        due = (~self.released[e]) & (self.release_t[e] <= self.elapsed[e])
        k = int(torch.where(due, self.release_t[e], torch.full_like(self.release_t[e], float("inf"))).argmin())
        if not self._put_down(env, e, k, float(self.lateral_y[e, k]), float(self.yaw[e, k]), step):
            return
        late = float(self.elapsed[e] - self.release_t[e, k])
        later = (~self.released[e]) & (self.release_t[e] > self.release_t[e, k])
        self.release_t[e, later] += late


    def _start_stream(self, e: int) -> None:
        """Continuous mode, at env ``e``'s reset: a resumed segment's state, or a new stream's first draw."""
        resume = self._resume.pop(e, None)
        if resume is None:
            p = self.cfg.params
            self._rng[e].manual_seed(int(p["seed"]) * 100003 + (int(self.episode[e]) - 1) * 7919 + e)
            self.tote_count[e] = 0
            self.segment[e] = 0
            self.segments_kept[e] = 0
            self.segments_discarded[e] = 0
            self._forced[e] = []
            self._draw_next(e, first=True)
            self._begin_segment(e, 0)
            print(f"[dexbench] L1 stream {int(self.episode[e])} (env {e}): continuous, a {p['lead']} first, the tote "
                  f"emptied every {self.episode_bags} bags", flush=True)
            return
        self.replaying[e] = True
        self._restore(e, resume)

    def _draw_next(self, e: int, *, first: bool = False) -> None:
        """Draw env ``e``'s next release: its kind (the lead first, then alternating while there are boxes), the belt's
        run to it, its lateral centre and its turn's draw (the item is chosen when it goes down)."""
        p, g = self.cfg.params, self._rng[e]
        alternate = self.episode_boxes > 0
        if first:
            self.next_kind[e] = _BOX if p["lead"] == "box" and alternate else _BAG
        elif alternate:
            self.next_kind[e] = 1 - int(self.next_kind[e])
        low, high = (float(v) for v in p["spacing_m"])
        gap = 0.0 if first else low + (high - low) * float(torch.rand(1, generator=g))
        self.next_t[e] = self.elapsed[e] + gap / float(p["belt_speed"])
        lo, hi = (float(v) for v in p["lateral_y"])
        lateral = lo + (hi - lo) * float(torch.rand(1, generator=g))
        unit, sign, flip = (float(v) for v in torch.rand(3, generator=g))
        self._next_draw[e] = (lateral, unit, sign < 0.5, flip < 0.5)

    def _choose(self, e: int, kind: int) -> int | None:
        """The item of ``kind`` to put down next in env ``e``: among the waiting ones, the bag placed fewest times in
        the session (a box: released fewest times), then the one released fewest times, then at random; None when none
        waits."""
        if kind == _BAG:
            waiting = [k for k in self.bags if int(self.state[e, k]) == WAITING and not bool(self.released[e, k])]
            keys = {k: (int(self.placed_total[e, k]), int(self.released_total[e, k])) for k in waiting}
        else:
            waiting = [k for k in self.boxes if not bool(self.box_live[e, k])]
            keys = {k: (int(self.released_total[e, k]),) for k in waiting}
        if not waiting:
            return None
        best = min(keys.values())
        ties = [k for k in waiting if keys[k] == best]
        return ties[int(torch.randint(len(ties), (1,), generator=self._rng[e]))] if len(ties) > 1 else ties[0]

    def _release_next(self, env, e: int, step: torch.Tensor) -> None:
        """Continuous mode: put env ``e``'s next item down if one waits and the belt is clear, then draw the next."""
        k = self._choose(e, int(self.next_kind[e]))
        if k is None:
            return
        lateral, unit, sign, flip = self._next_draw[e]
        if not self._put_down(env, e, k, lateral, self._turn(k, unit, sign, flip), step):
            return
        self.released_total[e, k] += 1
        self._event(e, "release", k, step)
        print(f"[dexbench] L1 stream (env {e}): {self._item_text(e, k)}", flush=True)
        self._draw_next(e)

    def _event(self, e: int, what: str, k: int, step: torch.Tensor) -> None:
        """Note a release, a take-off or a verdict in env ``e``'s running segment."""
        event = {"step": int(step[e]), "what": what, "item": k}
        if what == "release":
            event |= {"lateral_y_m": float(self.lateral_y[e, k]), "yaw_rad": float(self.yaw[e, k])}
        self._segment_events[e].append(event)

    def _snapshot(self, e: int, step: int) -> dict:
        """The stream's state in env ``e`` at a segment's start, JSON-ready and exact (float32 -> JSON -> float32)."""
        items = []
        for k, it in enumerate(self.items):
            zone = float(self.pick_zone_off_belt[e, k])
            items.append({
                "entity": it.entity, "variant": it.variant or None, "state": STATE_NAMES[int(self.state[e, k])],
                "released": bool(self.released[e, k]), "box_live": bool(self.box_live[e, k]),
                "lateral_y_m": float(self.lateral_y[e, k]), "yaw_rad": float(self.yaw[e, k]),
                "rest_s": float(self.rest_s[e, k]), "rest_tote": int(self.rest_tote[e, k]),
                "inside": float(self.inside[e, k]), "pick_zone_off_belt_rad": None if math.isnan(zone) else zone,
                "ref_y_m": float(self.ref_y[e, k]), "ref_yaw_rad": float(self.ref_yaw[e, k]),
                "release_step": int(self.release_step[e, k]), "final_step": int(self.final_step[e, k]),
                "placed_total": int(self.placed_total[e, k]), "released_total": int(self.released_total[e, k]),
            })
        return {"step": int(step), "elapsed_s": float(self.elapsed[e]), "tote_bags": int(self.tote_count[e]),
                "segment": int(self.segment[e]), "items": items}

    def _restore(self, e: int, segment: dict) -> None:
        """Env ``e``, just reset, as the stream stood at a recorded segment's start: the bags out made active again
        (the replay's ``reset_to`` writes their particles), the boxes out live, the judge's timers and the tote's load
        as they were. Its step counts on from the segment's start step."""
        start, dev = segment["start"], self._env.device
        self.step_offset[e] = int(start["step"])
        self.elapsed[e] = float(start["elapsed_s"])
        self.tote_count[e] = int(start["tote_bags"])
        self.segment[e] = int(start["segment"])
        self.plan_entry[e] = RESUMED
        names = {name: i for i, name in enumerate(STATE_NAMES)}
        e_ids = torch.tensor([e], device=dev)
        for k, item in enumerate(start["items"]):
            self.state[e, k] = names[item["state"]]
            self.released[e, k] = bool(item["released"])
            self.box_live[e, k] = bool(item["box_live"])
            self.lateral_y[e, k] = float(item["lateral_y_m"])
            self.yaw[e, k] = float(item["yaw_rad"])
            self.rest_s[e, k] = float(item["rest_s"])
            self.rest_tote[e, k] = int(item["rest_tote"])
            self.inside[e, k] = float(item["inside"])
            zone = item["pick_zone_off_belt_rad"]
            self.pick_zone_off_belt[e, k] = math.nan if zone is None else float(zone)
            self.ref_y[e, k] = float(item["ref_y_m"])
            self.ref_yaw[e, k] = float(item["ref_yaw_rad"])
            self.release_step[e, k] = int(item["release_step"])
            self.final_step[e, k] = int(item["final_step"])
            if self.items[k].kind == "bag" and int(self.state[e, k]) in (IN_PLAY, PLACED):
                self._env.scene[self.items[k].entity].write_active_to_sim_index(True, e_ids)
        self._segment_start[e] = start
        self._segment_events[e] = []
        print(f"[dexbench] L1 stream (env {e}): resumed at segment {int(self.segment[e])}, step {int(start['step'])}, "
              f"{len(self._forced[e])} releases to repeat", flush=True)

    def _begin_segment(self, e: int, step: int) -> None:
        self._segment_start[e] = self._snapshot(e, step)
        self._segment_events[e] = []

    def _close_segment(self, env, e: int, step: torch.Tensor, placed: list[int],
                       failed: list[tuple[int, str]]) -> None:
        """Continuous mode: env ``e``'s segment ends in this step: kept (exported) when a bag was placed and nothing
        failed, else discarded. Then the failed items wait again, a full tote is emptied and the next segment starts."""
        export = bool(self.cfg.params.get("export_segments")) and not self.replaying[e]
        manager = _active_recorder(env) if export else None
        kept = bool(placed) and not failed
        if kept:
            self._closing[e] = self._segment_record(e, int(step[e]), placed)
        if manager is not None:
            manager.record_pre_reset([e], force_export_or_skip=False)
            manager.set_success_to_episodes([e], torch.tensor([[kept]], dtype=torch.bool, device=env.device))
            manager.export_episodes([e])
        self._closing.pop(e, None)
        start = self._segment_start[e]["step"]
        if kept:
            self.segments_kept[e] += 1
            if self.replaying[e]:
                self.resumed_placed[e] = True
            print(f"[dexbench] L1 stream segment {int(self.segment[e])} (env {e}): placed "
                  f"{', '.join(self._label(k) for k in placed)} in steps {start + 1}-{int(step[e])}"
                  f"{', exported' if manager is not None else ''}", flush=True)
        else:
            self.segments_discarded[e] += 1
            why = ", ".join(f"{self._label(k)} {what}" for k, what in failed)
            print(f"[dexbench] L1 stream segment {int(self.segment[e])} (env {e}): discarded ({why}) in steps "
                  f"{start + 1}-{int(step[e])}", flush=True)
        for k, _ in failed:
            self._free(e, k)
        for k in placed if kept else ():
            self.placed_total[e, k] += 1
        self.tote_count[e] += len(placed)
        if int(self.tote_count[e]) >= self.episode_bags:
            in_tote = [k for k in self.bags if int(self.state[e, k]) == PLACED]
            e_ids = torch.tensor([e], device=env.device)
            for k in in_tote:
                self._park(k, e_ids)
                self._free(e, k)
            self.tote_count[e] = 0
            print(f"[dexbench] L1 stream (env {e}): the tote emptied ({len(in_tote)} bags)", flush=True)
        self.segment[e] += 1
        self._begin_segment(e, int(step[e]))
        if manager is not None:
            RecorderManager.record_post_reset(manager, [e])


    def __call__(self, env, env_ids, items, tote_boxes, belt_speed, belt_top_z, entry_x, clear_y, edge_margin,
                 lateral_y, spacing_m, bag_yaw_off_belt_deg, box_yaw_off_belt_deg, lead, miss_x, bag_clearance,
                 box_clearance, spawn_clearance, min_fraction, max_speed, hold_s, drop_z, missed_above_belt,
                 pick_zone_x, seed, disturbed_lateral, disturbed_yaw_deg, disturbed_lift, mode="episode",
                 episode_bags=None, episode_boxes=None, export_segments=False, plan=(), plan_start=0, plan_path="",
                 plan_sha256="", put_down=True, require_layout_draw=False, headline_bags=0) -> None:
        params = self.cfg.params
        dev = env.device
        dt = float(env.step_dt)
        belt_top = float(params["belt_top_z"])
        miss = float(params["miss_x"])
        drop = float(params["drop_z"])
        clear_lo, clear_hi = (float(v) for v in params["clear_y"])
        self.elapsed += dt
        step = env.episode_length_buf.to(dev) + self.step_offset
        failures: list[list[tuple[int, str]]] = [[] for _ in range(env.num_envs)]
        placed: list[list[int]] = [[] for _ in range(env.num_envs)]
        if self.continuous:
            for e in range(env.num_envs):
                if self.replaying[e]:
                    while self._forced[e] and self._forced[e][0][0] <= int(step[e]):
                        _, k, lateral, yaw = self._forced[e].pop(0)
                        self._put_down(env, e, k, lateral, yaw, step, check_clear=False)
                        self._event(e, "release", k, step)
                elif float(self.next_t[e]) <= float(self.elapsed[e]):
                    self._release_next(env, e, step)
        elif params.get("put_down", True):
            due = (~self.released) & (self.release_t <= self.elapsed[:, None])
            for e in due.any(dim=1).nonzero().flatten().tolist():
                self._release(env, e, step)
        self._judge_boxes(env, step, belt_top, miss, drop, clear_lo, clear_hi, failures)
        self._judge_bags(env, step, dt, belt_top, miss, drop, clear_lo, clear_hi, placed, failures)
        if self.continuous:
            for e in range(env.num_envs):
                if placed[e] or failures[e]:
                    self._close_segment(env, e, step, placed[e], failures[e])

    def _judge_boxes(self, env, step, belt_top, miss, drop, clear_lo, clear_hi, failures) -> None:
        """Each live box: disturbed once off the belt's carry (episode mode: flagged; continuous: taken off, its
        segment discarded); over the belt past the miss line, or fallen, taken off."""
        p = self.cfg.params
        limits = torch.tensor([float(p["disturbed_lateral"]), math.radians(float(p["disturbed_yaw_deg"])),
                               float(p["disturbed_lift"])], device=env.device)
        for k in self.boxes:
            e = self.box_live[:, k].nonzero().flatten()
            if len(e) == 0:
                continue
            pos, yaw = self._box_pose(k, e)
            off = torch.stack([(pos[:, 1] - self.ref_y[e, k]).abs(),
                               torch.remainder(yaw - self.ref_yaw[e, k] + math.pi, 2.0 * math.pi).sub(math.pi).abs(),
                               (pos[:, 2] - belt_top).clamp(min=0.0)], dim=-1)
            judged = self.release_step[e, k] < step[e]
            self.deviation[e, k] = torch.where(judged[:, None], torch.maximum(self.deviation[e, k], off),
                                               self.deviation[e, k])
            beyond = off > limits
            newly = judged & ~self.disturbed[e, k] & beyond.any(dim=-1)
            if newly.any():
                self.disturbed[e[newly], k] = True
                self.disturbed_step[e[newly], k] = step[e[newly]]
                for row in newly.nonzero().flatten().tolist():
                    i = int(e[row])
                    what = "disturbed (" + ", ".join(
                        f"{name} {value:.3g}" for name, value, over in zip(
                            ("across m", "turned deg", "lifted m"),
                            (float(off[row, 0]), math.degrees(float(off[row, 1])), float(off[row, 2])),
                            beyond[row].tolist()) if over) + ")"
                    if self.continuous:
                        failures[i].append((k, what))
                        self._event(i, "disturbed", k, step)
                    print(f"[dexbench] L1 stream (env {i}): {self._label(k)} {what}", flush=True)
            over_belt = (pos[:, 1] >= clear_lo) & (pos[:, 1] <= clear_hi)
            _, hi_off = self._extent_xy(k, yaw)
            gone = (over_belt & (pos[:, 0] + hi_off[:, 0] >= miss)) | (pos[:, 2] < drop)
            if self.continuous:
                gone |= self.disturbed[e, k]
            if gone.any():
                self._park(k, e[gone])
                self.final_step[e[gone], k] = step[e[gone]]
                if self.continuous:
                    for i in e[gone].tolist():
                        self._event(i, "taken_off", k, step)
                        self._free(i, k)

    def _judge_bags(self, env, step, dt, belt_top, miss, drop, clear_lo, clear_hi, placed, failures) -> None:
        params = self.cfg.params
        origins = env.scene.env_origins
        for k in self.bags:
            live = (self.state[:, k] == IN_PLAY).nonzero().flatten()
            if len(live) == 0:
                continue
            bag = env.scene[self.items[k].entity]
            pts = bag.data.particle_pos_w.torch[live] - origins[live][:, None, :]
            speed_mean = bag.data.particle_vel_w.torch[live].norm(dim=-1).mean(dim=1)
            in_box = (pts[:, :, None, :] >= self.tote_lo) & (pts[:, :, None, :] <= self.tote_hi)
            inside = in_box.all(-1).float().mean(1)
            frac, tote = inside.max(dim=1)
            self.inside[live, k] = frac
            resting = (frac >= float(params["min_fraction"])) & (speed_mean < float(params["max_speed"]))
            same = resting & (self.rest_tote[live, k] == tote)
            self.rest_s[live, k] = torch.where(same, self.rest_s[live, k] + dt,
                                               torch.where(resting, torch.full_like(frac, dt), torch.zeros_like(frac)))
            self.rest_tote[live, k] = torch.where(resting, tote, torch.full_like(tote, -1))
            done = self.rest_s[live, k] >= float(params["hold_s"])
            target = self.items[k].target_tote
            self.state[live[done], k] = torch.where(tote[done] == target, torch.full_like(tote[done], PLACED),
                                                    torch.full_like(tote[done], WRONG_TOTE))
            centre = pts.mean(dim=1)
            lowest = pts[..., 2].min(dim=1).values
            dropped = (~done) & (centre[:, 2] < drop) & (frac < 0.05)
            self.state[live[dropped], k] = DROPPED
            on_belt = ((lowest < belt_top + float(params["missed_above_belt"])) & (centre[:, 1] >= clear_lo)
                       & (centre[:, 1] <= clear_hi))
            missed = (~done) & (~dropped) & on_belt & (pts[..., 0].max(dim=1).values >= miss)
            self.state[live[missed], k] = MISSED
            final = done | dropped | missed
            self.final_step[live[final], k] = step[live[final]]
            first = (torch.isnan(self.pick_zone_off_belt[live, k]) & on_belt
                     & (centre[:, 0] >= float(params["pick_zone_x"])))
            if first.any():
                angle = torch.remainder(long_axis_angle(pts[first, :, :2]), math.pi)
                self.pick_zone_off_belt[live[first], k] = torch.minimum(angle, math.pi - angle)
            away = missed
            if self.continuous:
                away = away | dropped | (done & (self.state[live, k] == WRONG_TOTE))
            if away.any():
                self._park(k, live[away])
            if not self.continuous:
                continue
            for i, state in zip(live[final].tolist(), self.state[live[final], k].tolist()):
                self._event(i, STATE_NAMES[state], k, step)
                if state == PLACED:
                    placed[i].append(k)
                else:
                    failures[i].append((k, STATE_NAMES[state]))


    def bag_states(self) -> torch.Tensor:
        return self.state[:, self.bags]

    def succeeded(self) -> torch.Tensor:
        """Episode mode: every bag of the episode placed and no box disturbed (DB-L1's challenge track; its whole
        success without ``headline_bags``, which a collection plan moves on by). Continuous mode: only in a resumed
        segment (a replay of a demo, ``schedule_resume``), once its bag is placed again and nothing failed, so that a
        replay tool judges the demo as its recording did; a live stream never succeeds (every placed bag is a demo
        of its own)."""
        if self.continuous:
            return self.resumed_placed.clone()
        bags = (self.bag_states() == PLACED) | ~self.used[:, self.bags]
        return bags.all(dim=1) & ~self.disturbed.any(dim=1)

    def _judged(self, envs: list[int]) -> list[dict]:
        """Envs ``envs``' episodes as ``polybag_stream_record.placed_in_order_steps`` reads them: the items in planned
        release order, with their states and steps (stream steps; an episode's start at 0). One copy to the host per
        tensor, whatever the number of envs."""
        planned, states, final = self.planned_release_t.tolist(), self.state.tolist(), self.final_step.tolist()
        used, disturbed, disturbed_step = self.used.tolist(), self.disturbed.tolist(), self.disturbed_step.tolist()
        return [{"order": [k for k in sorted(range(len(self.items)), key=lambda k: (planned[e][k], k))
                           if math.isfinite(planned[e][k])],
                 "items": [{"kind": it.kind, "used": used[e][k], "state": STATE_NAMES[states[e][k]],
                            "final_step": final[e][k], "disturbed": disturbed[e][k], "disturbed_step": disturbed_step[e][k]}
                           for k, it in enumerate(self.items)]} for e in envs]

    def bags_placed_in_order(self, envs: torch.Tensor | None = None) -> torch.Tensor:
        """Per env (episode mode): how many of its bags, counted in stream order from the first, were placed before
        the episode's first failure (``polybag_stream_record.placed_in_order_steps``); 0 in continuous mode, and for
        an env outside ``envs`` (a mask; None: every env)."""
        counts = torch.zeros(self._env.num_envs, dtype=torch.long, device=self._env.device)
        if self.continuous:
            return counts
        chosen = list(range(self._env.num_envs)) if envs is None else torch.nonzero(envs).flatten().tolist()
        if chosen:
            counts[chosen] = torch.tensor([len(placed_in_order_steps(judged)) for judged in self._judged(chosen)],
                                          dtype=torch.long, device=self._env.device)
        return counts

    def headline_succeeded(self, envs: torch.Tensor | None = None) -> torch.Tensor:
        """Episode mode with ``headline_bags``: its first ``headline_bags`` bags placed in stream order before any
        failure, judged for the envs of ``envs`` (a mask; None: every env) and false for the others. Latched by
        construction: a bag's final state and a box's disturbance never change within an episode, so a later failure
        cannot undo it. False everywhere without ``headline_bags``."""
        if not self.headline_bags:
            return torch.zeros(self._env.num_envs, dtype=torch.bool, device=self._env.device)
        return self.bags_placed_in_order(envs) >= self.headline_bags

    def failed(self) -> torch.Tensor:
        """Episode mode: a bag of the episode dropped, missed or in the wrong tote, or a box disturbed. Never in
        continuous mode, where a failure discards a segment and the stream goes on."""
        if self.continuous:
            return torch.zeros(self._env.num_envs, dtype=torch.bool, device=self._env.device)
        states = self.bag_states()
        bad = torch.stack([states == f for f in FAILED_STATES]).any(0) & self.used[:, self.bags]
        return bad.any(dim=1) | self.disturbed.any(dim=1)

    def summary(self, env_id: int = 0) -> dict[str, int]:
        used = self.used[env_id, self.bags].tolist()
        s = [state for state, u in zip(self.bag_states()[env_id].tolist(), used) if u]
        out = {name: s.count(i) for i, name in enumerate(STATE_NAMES)}
        out["boxes disturbed"] = int(self.disturbed[env_id].sum())
        return out

    def _session_record(self) -> dict:
        p = self.cfg.params
        return {
            "mode": self.mode,
            "belt_speed_m_s": float(p["belt_speed"]),
            "seed": int(p["seed"]),
            "lead": p["lead"],
            "episode_bags": self.episode_bags,
            "episode_boxes": self.episode_boxes,
            "spacing_range_m": [float(v) for v in p["spacing_m"]],
            "lateral_y_range_m": [float(v) for v in p["lateral_y"]],
            "clear_y_m": [float(v) for v in p["clear_y"]],
            "bag_yaw_off_belt_deg": [float(v) for v in p["bag_yaw_off_belt_deg"]],
            "box_yaw_off_belt_deg": [float(v) for v in p["box_yaw_off_belt_deg"]],
            "entry_x_m": float(p["entry_x"]),
            "judge": {"min_fraction": float(p["min_fraction"]), "max_speed_m_s": float(p["max_speed"]),
                      "hold_s": float(p["hold_s"]), "drop_z_m": float(p["drop_z"]), "miss_x_m": float(p["miss_x"]),
                      "box_disturbed": {"across_m": float(p["disturbed_lateral"]),
                                        "turned_deg": float(p["disturbed_yaw_deg"]),
                                        "lifted_m": float(p["disturbed_lift"])}},
        }

    def episode_record(self, env_id: int = 0) -> dict:
        """Env ``env_id``'s episode so far (episode mode), or the segment being exported (continuous mode; the
        running one between exports), JSON-ready (see the module docstring); read it before the env resets."""
        e = int(env_id)
        if self.continuous:
            return self._closing[e] if e in self._closing else self._segment_record(
                e, int(self._env.episode_length_buf[e] + self.step_offset[e]), [])
        p = self.cfg.params
        entry = int(self.plan_entry[e])
        if entry >= 0:
            plan = {"source": "plan", "path": p.get("plan_path") or None, "sha256": p.get("plan_sha256") or None,
                    "entry": entry, "entries": len(self.plan), "id": self.plan[entry].id,
                    "attempt": int(self.plan_attempt[e])}
        else:
            plan = {"source": {SCHEDULED: "scheduled", LAID: "variation"}.get(entry, "drawn")}
        planned = self.planned_release_t[e].tolist()
        used = self.used[e, self.bags]
        return {
            "version": EPISODE_RECORD_VERSION,
            "mode": self.mode,
            "episode": int(self.episode[e]),
            "steps": int(self._env.episode_length_buf[e]),
            "step_dt": float(self._env.step_dt),
            "elapsed_s": round(float(self.elapsed[e]), 4),
            "success": bool(self.succeeded()[e]),
            "failed": bool(self.failed()[e]),
            "stream_over": bool(self.succeeded()[e] | self.failed()[e]),
            "placed": int(((self.bag_states()[e] == PLACED) & used).sum()),
            "bags": int(used.sum()),
            "session": self._session_record(),
            "plan": plan,
            "order": [k for k in sorted(range(len(self.items)), key=lambda k: (planned[k], k))
                      if math.isfinite(planned[k])],
            "items": [self._item_record(e, k, 0) for k in range(len(self.items))],
        }

    def _segment_record(self, e: int, step: int, placed: list[int]) -> dict:
        """Env ``e``'s running segment of a continuous stream, ending in stream step ``step`` with ``placed`` (its
        bags placed in that step): its record, with the stream's state at its start and its events."""
        start = self._segment_start[e]
        first = int(start["step"])
        return {
            "version": EPISODE_RECORD_VERSION,
            "mode": self.mode,
            "episode": int(self.episode[e]),
            "steps": step - first,
            "step_dt": float(self._env.step_dt),
            "elapsed_s": round(float(self.elapsed[e]) - float(start["elapsed_s"]), 4),
            "success": bool(placed),
            "failed": False,
            "stream_over": bool(placed),
            "placed": len(placed),
            "bags": len(placed),
            "session": {**self._session_record(), "tote_emptied_every": self.episode_bags},
            "plan": {"source": "drawn"},
            "segment": {"index": int(self.segment[e]), "start_step": first, "end_step": step,
                        "placed": [self.items[k].variant or self.items[k].entity for k in placed],
                        "tote_bags_at_start": int(start["tote_bags"]),
                        "start": start, "events": list(self._segment_events[e])},
            "grasp_style": None,
            "order": [ev["item"] for ev in self._segment_events[e] if ev["what"] == "release"],
            "items": [self._item_record(e, k, first) for k in range(len(self.items))],
        }

    def _item_record(self, e: int, k: int, first: int) -> dict:
        """Item ``k`` of env ``e``, its steps counted from stream step ``first`` (an earlier one as 0)."""
        it = self.items[k]
        yaw_deg = math.degrees(float(self.yaw[e, k]))

        def since(s: int) -> int:
            return -1 if s < 0 else max(s - first, 0)

        planned = float(self.planned_release_t[e, k])
        out = bool(self.released[e, k])
        record = {"entity": it.entity, "kind": it.kind, "variant": it.variant or None, "used": bool(self.used[e, k]),
                  "release_t_planned_s": planned if math.isfinite(planned) else None,
                  "lateral_y_m": float(self.lateral_y[e, k]),
                  "yaw_deg": yaw_deg,
                  "long_axis_deg": round(math.degrees(float(self.rest_long_axis[k])) + yaw_deg, 3) % 180.0,
                  "off_belt_deg": round(self._off_belt_deg(e, k), 3),
                  "release_step": since(int(self.release_step[e, k])) if out else -1,
                  "final_step": since(int(self.final_step[e, k])) if out else -1}
        hx, hy = it.half_extent_xy
        if it.kind == "box":
            state = "live" if bool(self.box_live[e, k]) else "parked" if out else "waiting"
            dev = self.deviation[e, k].tolist()
            record |= {"state": state, "size_m": [2 * hx, 2 * hy, it.height],
                       "disturbed": bool(self.disturbed[e, k]),
                       "disturbed_step": since(int(self.disturbed_step[e, k])),
                       "max_across_m": round(dev[0], 4), "max_turned_deg": round(math.degrees(dev[1]), 3),
                       "max_lifted_m": round(dev[2], 4)}
            box = BOX_VARIANTS_BY_KEY.get(it.variant)
            if box is not None:
                record |= {"box_number": box.number, "sort_code": box.sort_code}
            return record
        at_pick = float(self.pick_zone_off_belt[e, k])
        record |= {"off_belt_at_pick_zone_deg": None if math.isnan(at_pick) else round(math.degrees(at_pick), 3),
                   "state": STATE_NAMES[int(self.state[e, k])], "inside": round(float(self.inside[e, k]), 3),
                   "target_tote": it.target_tote, "footprint_m": [2 * hx, 2 * hy]}
        bag = SORT_VARIANTS_BY_KEY.get(it.variant)
        if bag is not None:
            record |= {"body": bag.body.key, "colour": bag.body.colour_name, "film_srgb": list(bag.body.rgb),
                       "nominal_cm": list(bag.body.nominal_cm), "thickness_m": bag.thickness_m, "split": bag.split,
                       "sort_code": bag.sort_code}
        return record


STREAM_TERM = "l1_stream"
"""The name of the stream's event term in the task's events cfg."""


def l1_stream(env) -> PolybagStream:
    """The env's stream: the instance Isaac Lab's event manager made of the ``STREAM_TERM`` term. Raises (the
    manager's ValueError) when the env has no such term, so a renamed term fails at its first read instead of the
    episode never succeeding, and RuntimeError before the manager has made it (it does at the simulation's play)."""
    stream = env.event_manager.get_term_cfg(STREAM_TERM).func
    if isinstance(stream, type):
        raise RuntimeError(f"the {STREAM_TERM!r} event term is not made yet (Isaac Lab makes it when the simulation "
                           "plays)")
    return stream


def stream_success(env) -> torch.Tensor:
    """Success (episode mode): every bag of the episode is in its tote and no box was disturbed. In continuous mode
    only a resumed segment's bag placed again (a replay's), never a live stream's (``PolybagStream.succeeded``)."""
    stream = l1_stream(env)
    if not stream.headline_bags:
        return stream.succeeded()
    over = stream.succeeded() | stream.failed() | time_out(env)
    if not bool(over.any()):
        return over
    return stream.headline_succeeded(over) & over


def stream_failed(env) -> torch.Tensor:
    """The episode's first failure (episode mode): a bag dropped, missed or in the wrong tote, or a box disturbed;
    false whenever ``stream_success`` is true (with ``headline_bags``, a failure after the headline's bags were placed
    ends the episode as a success). Never in continuous mode."""
    stream = l1_stream(env)
    failed = stream.failed()
    if not stream.headline_bags:
        return failed & ~stream.succeeded()
    if not bool(failed.any()):
        return failed
    return failed & ~stream.headline_succeeded(failed)


def record_stream_tracks(env, env_id: int) -> dict:
    """Episode recorder term (DB-L1's environment): the episode's ``bags_placed_in_order`` and its
    ``challenge_success`` (every bag placed and no box disturbed, ``PolybagStream.succeeded``), beside the success
    Arena records from the ``success`` term (the headline rule under ``headline_bags``); the scorer reports the
    challenge track from it. Nothing for a continuous stream, whose segments are its demos."""
    stream = l1_stream(env)
    if stream.continuous:
        return {}
    only = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    only[env_id] = True
    return {"bags_placed_in_order": int(stream.bags_placed_in_order(only)[env_id]),
            "challenge_success": bool(stream.succeeded()[env_id])}
