# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Pick-the-polybags-off-the-conveyor task for the DexBench E2 logistics centre (DB-L1)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass
from pathlib import Path
from typing import Any

import h5py

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

from dexbench_arena.assets.polybag.polybag_object_cfg import PolybagObjectCfg, PolybagVisualCfg
from dexbench_arena.assets.polybag.template import TET_MATERIAL_FIELDS, load_cfg_template
from dexbench_arena.contact_materials import HandContactMaterialCfg
from dexbench_arena.tasks.polybag_stream import (
    PolybagStream,
    StreamItem,
    l1_stream,
    stream_failed,
    stream_success,
)
from dexbench_arena.tasks.polybag_stream_record import (
    EPISODE_RECORD_ATTR,
    drawn_on,
    recorded_layout,
    reused_seed_refusal,
)
from dexbench_arena.tasks.recording_hooks import DexBenchRecordingTask
from dexbench_arena.tasks.task_termination import task_termination_cfg

IDENTITY_XYZW = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class L1Bag:
    """One bag of the stream: its scene entity, its pool variant and package, and where it waits."""

    entity: str
    variant: str
    usd_path: str
    footprint_m: tuple[float, float]
    """Its nominal footprint (x, y) at its rest pose [m] (``sort_variants.BagBody.footprint_m``)."""
    park_xy: tuple[float, float]
    """Its park slot's centre (environment frame); its rest bounds are centred there."""


@configclass
class PolybagEventsCfg:
    """The term run every control step: the item stream (``tasks.polybag_stream``)."""

    l1_stream: EventTermCfg = MISSING


@configclass
class PolybagStreamTerminationsCfg:
    """Success when every bag of the episode is in the tote and no box was disturbed; the failure ``stream_failed``
    on the first bag dropped, missed or in the wrong tote, or box disturbed. The two are never true together. With
    the stream's ``headline_bags`` the success is the headline rule, true when the episode ends as above (every bag
    placed, the first failure or the cap) after its first ``headline_bags`` bags were placed in stream order, so the
    episode ends where the all-bags rule ends it and a failure after the headline is not one (``stream_success``). A continuous
    stream fails never and succeeds only in a replay of one of its demos, once the demo's bag is placed again (so a
    replay tool such as rerecord_demos.py judges it as the recording did). The judging is the stream's interval event, after the step's termination check,
    so the episode ends on the control step after the verdict. The episode limit is the task's ``timeout_s``."""

    success: TerminationTermCfg = TerminationTermCfg(func=stream_success)

    stream_failed: TerminationTermCfg = TerminationTermCfg(func=stream_failed)


def polybag_scene_cfg(bags: Sequence[tuple[str, PolybagObjectCfg]]) -> Any:
    """The task's own scene entities, one field per bag; Arena merges a task's scene cfg field by field, so the class
    is made for the bags it holds."""

    names = [name for name, _ in bags]
    if len(set(names)) != len(names):
        raise ValueError(f"polybag names repeat: {names}")
    namespace = {"__annotations__": dict.fromkeys(names, PolybagObjectCfg), **dict.fromkeys(names, MISSING),
                 "__doc__": "The task's VBD polybags, one scene entity each."}
    cls = configclass(type("PickPolybagFromConveyorSceneCfg", (), namespace))
    return cls(**dict(bags))


class DexBenchPickPolybagFromConveyorTask(DexBenchRecordingTask):
    """Pick every polybag of the item stream off the belt and put it in the tote; let the boxes pass (DB-L1)."""

    def __init__(
        self,
        bags: Sequence[L1Bag],
        boxes: Sequence[StreamItem],
        stream: Mapping[str, Any],
        *,
        park_surface_z: float,
        park_clearance: float,
        vbd_substeps: int,
        vbd_iterations: int,
        mjwarp_substeps: int,
        full_surface_contact: bool,
        frame_truncation: str | None,
        coupling_mode: str,
        material: Mapping[str, float | None],
        episode_length_s: float = 600.0,
        task_description: str | None = None,
        robot_joint_pos_overrides: dict[str, float] | None = None,
        hand_contact_material: HandContactMaterialCfg | None = None,
        viewer_eye_lookat: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
        box_colliders: Sequence[str] = (),
        contact_log_barrier: bool = False,
    ):
        super().__init__(
            episode_length_s=episode_length_s,
            task_description=task_description
            or "Pick every polybag off the conveyor and put it in the tote; let the boxes pass.",
        )
        if frame_truncation not in ("init", "all", None):
            raise ValueError(f"frame_truncation {frame_truncation!r}: one of 'init', 'all', None")
        if coupling_mode not in ("one_way", "two_way"):
            raise ValueError(f"coupling_mode {coupling_mode!r}: one of 'one_way', 'two_way'")
        unknown = set(material) - set(TET_MATERIAL_FIELDS)
        if unknown:
            raise TypeError(f"material: unknown fields {sorted(unknown)}; the tet bag takes {TET_MATERIAL_FIELDS}")
        if not bags:
            raise ValueError("pick_polybag_from_conveyor: the stream needs a bag")
        self.hand_contact_material = hand_contact_material
        self.viewer_eye_lookat = viewer_eye_lookat
        self.robot_joint_pos_overrides = dict(robot_joint_pos_overrides) if robot_joint_pos_overrides else None
        self.material = dict(material)
        self.stepping = {"vbd_substeps": int(vbd_substeps), "vbd_iterations": int(vbd_iterations),
                         "mjwarp_substeps": int(mjwarp_substeps), "full_surface_contact": bool(full_surface_contact),
                         "frame_truncation": frame_truncation, "coupling_mode": coupling_mode,
                         "contact_log_barrier": bool(contact_log_barrier)}
        fields, bag_items = [], []
        for bag in bags:
            cfg = PolybagObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{bag.entity}",
                spawn=PolybagVisualCfg(usd_path=str(bag.usd_path)),
                cotton_youngs_modulus=None,
                cotton_poisson_ratio=None,
                film_rest_bend_angles=None,
                box_colliders=tuple(box_colliders),
                **self.material,
                **self.stepping,
            )
            rest = load_cfg_template(cfg).rest
            lo, hi = rest.min(axis=0), rest.max(axis=0)
            park = (float(bag.park_xy[0] - (lo[0] + hi[0]) / 2.0), float(bag.park_xy[1] - (lo[1] + hi[1]) / 2.0),
                    float(park_surface_z + park_clearance - lo[2]))
            cfg.init_state = PolybagObjectCfg.InitialStateCfg(pos=park, rot=IDENTITY_XYZW)
            fields.append((bag.entity, cfg))
            fx, fy = bag.footprint_m
            bag_items.append(StreamItem("bag", bag.entity, (fx / 2.0, fy / 2.0), (*park, *IDENTITY_XYZW), 0, 0.0,
                                        bag.variant))
        self.scene_cfg = polybag_scene_cfg(fields)
        params = {**dict(stream), "items": (*bag_items, *boxes)}
        self.events_cfg = PolybagEventsCfg(
            l1_stream=EventTermCfg(func=PolybagStream, mode="interval", interval_range_s=(0.0, 0.0),
                                   is_global_time=True, params=params))
        self.termination_cfg = PolybagStreamTerminationsCfg()

    def get_scene_cfg(self) -> Any:
        return self.scene_cfg

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return task_termination_cfg(self.episode_length_s, self.termination_cfg)

    def get_viewer_cfg(self) -> ViewerCfg:
        if self.viewer_eye_lookat is None:
            return super().get_viewer_cfg()
        eye, lookat = self.viewer_eye_lookat
        return ViewerCfg(eye=tuple(eye), lookat=tuple(lookat))

    def get_events_cfg(self) -> Any:
        return self.events_cfg

    def episode_attributes(self, env, env_id: int) -> dict[str, str]:
        """The attributes a recorded demo of env ``env_id``'s episode (or of the stream's segment being exported)
        carries, read as it is written: ``dexbench_l1_stream``, the stream's ``episode_record`` with the bags'
        stepping and material (``tet_*``, null for the asset's) in its session, as JSON."""
        record = l1_stream(env).episode_record(env_id)
        record["session"]["polybag_stepping"] = dict(self.stepping)
        record["session"]["polybag_material"] = dict(self.material)
        return {EPISODE_RECORD_ATTR: json.dumps(record, allow_nan=False)}

    def validate_recording_dataset(self, path) -> None:
        """Before record_demos.py records into ``path`` (its ``--dataset_file``): ValueError when a recording in the
        same directory, ``path`` itself included, holds a demo whose episode was drawn on this session's stream seed
        (its ``dexbench_l1_stream`` record): ``polybag_stream_record.reused_seed_refusal``, whose message names the
        next seed none of them drew on (the collection console suggests seeds by the same rule). A file that cannot
        be read is reported and skipped."""
        seed = int(self.events_cfg.l1_stream.params["seed"])
        drawn: dict[int, list[str]] = {}
        for file in sorted(Path(path).parent.glob("*.hdf5")):
            try:
                with h5py.File(file, "r") as recording:
                    data = recording.get("data")
                    for name, demo in ({} if data is None else data).items():
                        text = demo.attrs.get(EPISODE_RECORD_ATTR)
                        drawn_seed = drawn_on({} if text is None else json.loads(text))
                        if drawn_seed is not None:
                            drawn.setdefault(drawn_seed, []).append(f"{file.name}:{name}")
            except (OSError, KeyError, ValueError) as error:
                print(f"[dexbench] DB-L1: {file} was not read for its stream seeds ({type(error).__name__}: {error})",
                      flush=True)
        refusal = reused_seed_refusal(seed, drawn, Path(path).parent)
        if refusal is not None:
            raise ValueError(refusal)

    def prepare_replay_episode(self, env, demo_attributes: Mapping[str, Any]) -> None:
        """Before a replay resets env 0 to a recorded demo's initial state (``rerecord_demos.py``, ahead of
        ``reset_to``): that reset lays the stream out as the demo's episode was, from its ``dexbench_l1_stream`` record
        (``polybag_stream_record.recorded_layout`` into ``PolybagStream.schedule_next``), or, for a continuous stream's
        demo, resumes the stream as it stood at the segment's start with its releases at the recorded steps
        (``PolybagStream.schedule_resume``). Without it the reset draws from the replay process's own episode count,
        which is not the recording's (the replay resets once before its first demo, and the recording counted every
        discarded try). The recorded state restores where the items are, not when and where the rest go down.
        ValueError for a demo without the record or one of another stream (or of the other mode)."""
        text = demo_attributes.get(EPISODE_RECORD_ATTR)
        if text is None:
            raise ValueError(f"pick_polybag_from_conveyor: the demo carries no {EPISODE_RECORD_ATTR} attribute, so its "
                             "stream layout cannot be replayed (a replay would draw another one)")
        stream, record = l1_stream(env), json.loads(text)
        if record.get("mode", "episode") == "continuous":
            stream.schedule_resume(0, record["segment"])
            return
        stream.schedule_next(0, *recorded_layout(record, [(it.entity, it.variant) for it in stream.items]))

    def get_mimic_env_cfg(self, arm_mode: Any) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]
