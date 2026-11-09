# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Read a db-m5-create-stable-clutter (dbclutter) report as DB-M5's parts-bin pile."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import hashlib
import json
import math
import re
from pathlib import Path

from dexbench_arena.layouts.db_m5_parts import DEFAULT_PART, get_m5_part

SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3, 4, 5)
QUATERNION_NORM_TOLERANCE = 1e-3
BIN_GEOMETRY_TOLERANCE_M = 1e-6

Pose7 = tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class PartsBinClutter:
    part: str
    """The ``db_m5_parts`` key of every part in the pile."""

    poses_bin_local: tuple[Pose7, ...]
    """``(px, py, pz, qx, qy, qz, qw)`` of each part's root, in the bin's frame, by report index."""

    path: str
    sha256: str
    run_tag: str | None
    source_commit: str | None
    """dbclutter's ``versions.git_commit``; with ``source_dirty`` it does not identify the generator."""

    source_dirty: bool | None


def _refuse(path: Path, message: str) -> ValueError:
    return ValueError(f"{path}: {message}")


def _either(block: dict, new: str, old: str):
    """``block[new]``, or ``block[old]`` for a report from before schema 5."""

    return block[new] if new in block else block.get(old)


def load_parts_bin_clutter(
    path: str | Path,
    *,
    bin_interior_half_xy: tuple[float, float],
    bin_floor_top_z: float,
) -> PartsBinClutter:
    """Load and check the report at ``path`` against DB-M5's bin geometry."""

    path = Path(path).expanduser().resolve()
    data = path.read_bytes()
    return _load(str(path), data, tuple(bin_interior_half_xy), float(bin_floor_top_z))


@lru_cache(maxsize=64)
def _load(path_text: str, data: bytes, bin_interior_half_xy, bin_floor_top_z) -> PartsBinClutter:
    path = Path(path_text)
    try:
        report = json.loads(data)
    except json.JSONDecodeError as exc:
        raise _refuse(path, f"not JSON ({exc})") from None
    if not isinstance(report, dict):
        raise _refuse(path, "not a dbclutter clutter report")

    schema = report.get("schema_version")
    if schema not in SUPPORTED_SCHEMA_VERSIONS:
        raise _refuse(path, f"schema_version {schema!r}; this reads {SUPPORTED_SCHEMA_VERSIONS}")

    request = report.get("request") or {}
    part = request.get("part", DEFAULT_PART)
    try:
        get_m5_part(part)
    except ValueError as exc:
        raise _refuse(path, f"request.part: {exc}") from None

    scene = report.get("scene") or {}
    if scene.get("bin_at_origin") is not True:
        raise _refuse(path, "scene.bin_at_origin is not true; the poses are not in the bin's frame")
    half_xy = scene.get("bin_interior_half_xy")
    floor_z = scene.get("bin_floor_top_z")
    if (
        not isinstance(half_xy, list)
        or len(half_xy) != 2
        or not all(math.isclose(a, b, abs_tol=BIN_GEOMETRY_TOLERANCE_M) for a, b in zip(half_xy, bin_interior_half_xy))
    ):
        raise _refuse(path, f"scene.bin_interior_half_xy {half_xy!r}, DB-M5's bin is {list(bin_interior_half_xy)}")
    if not isinstance(floor_z, (int, float)) or not math.isclose(floor_z, bin_floor_top_z, abs_tol=BIN_GEOMETRY_TOLERANCE_M):
        raise _refuse(path, f"scene.bin_floor_top_z {floor_z!r}, DB-M5's bin is {bin_floor_top_z}")

    verdict = report.get("verdict") or {}
    entries = _either(verdict, "parts", "bolts")
    if not isinstance(entries, list) or not entries:
        raise _refuse(path, "verdict.parts lists no parts")
    indexed: dict[int, Pose7] = {}
    for position, entry in enumerate(entries):
        index = _either(entry, "index", "bolt")
        where = f"verdict.parts[{position}]"
        if not isinstance(index, int) or isinstance(index, bool):
            raise _refuse(path, f"{where} has no index")
        if index in indexed:
            raise _refuse(path, f"{where}: index {index} appears twice")
        outside = [key for key in ("escaped", "fell_through") if entry.get(key)]
        if entry.get("contained") is False:
            outside.insert(0, "not contained")
        if outside:
            raise _refuse(path, f"{where} (index {index}) is outside the bin: {', '.join(outside)}")
        pose = entry.get("pose_exact")
        if (
            not isinstance(pose, list)
            or len(pose) != 7
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in pose)
        ):
            raise _refuse(path, f"{where}.pose_exact is not 7 finite numbers: {pose!r}")
        norm = math.sqrt(sum(v * v for v in pose[3:]))
        if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
            raise _refuse(path, f"{where}.pose_exact quaternion has norm {norm:.6f}")
        indexed[index] = tuple(float(v) for v in pose)

    if verdict.get("converged") is False:
        print(f"[dexbench] WARNING: {path}: dbclutter reports the pile did not converge (verdict.converged false)")
    versions = report.get("versions") or {}
    if versions.get("git_dirty"):
        print(
            f"[dexbench] WARNING: {path}: made from an uncommitted dbclutter tree "
            f"(versions.git_dirty); git_commit {versions.get('git_commit')} alone does not reproduce it"
        )
    return PartsBinClutter(
        part=part,
        poses_bin_local=tuple(indexed[index] for index in sorted(indexed)),
        path=path_text,
        sha256=hashlib.sha256(data).hexdigest(),
        run_tag=report.get("run_tag"),
        source_commit=versions.get("git_commit"),
        source_dirty=versions.get("git_dirty"),
    )


@dataclass(frozen=True)
class PartsBinPileSet:
    """Several settled piles of one part, by dbclutter seed: the piles an environment draws among."""

    part: str
    piles: tuple[tuple[int, PartsBinClutter], ...]
    """``(seed, pile)`` by ascending seed; every pile has the same number of parts."""

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(seed for seed, _ in self.piles)

    @property
    def sha256(self) -> str:
        """One hash of the whole set: of each ``seed:report sha256`` line, by seed."""

        return hashlib.sha256("".join(f"{seed}:{pile.sha256}\n" for seed, pile in self.piles).encode()).hexdigest()


def load_parts_bin_pile_set(
    reports: dict[int, str | Path],
    *,
    part: str,
    bin_interior_half_xy: tuple[float, float],
    bin_floor_top_z: float,
) -> PartsBinPileSet:
    """Load ``{seed: report}`` as one part's pile set; ValueError for a report of another part or another pile size."""

    if not reports:
        raise ValueError(f"no {part} piles to load")
    piles = []
    for seed in sorted(reports):
        pile = load_parts_bin_clutter(reports[seed], bin_interior_half_xy=bin_interior_half_xy, bin_floor_top_z=bin_floor_top_z)
        if pile.part != part:
            raise _refuse(Path(pile.path), f"a {pile.part} pile, in a set of {part} piles")
        if piles and len(pile.poses_bin_local) != len(piles[0][1].poses_bin_local):
            raise _refuse(
                Path(pile.path),
                f"{len(pile.poses_bin_local)} parts; seed {piles[0][0]}'s pile has {len(piles[0][1].poses_bin_local)}, "
                "and the scene spawns one set of parts for every pile",
            )
        piles.append((int(seed), pile))
    return PartsBinPileSet(part=part, piles=tuple(piles))


_SEED_IN_NAME = re.compile(r"_seed(\d+)_")


def seed_reports(part_dir: Path) -> dict[int, Path]:
    """``{seed: report}`` for one part's directory (``<set>/<part>``); the first report found wins a seed."""

    part_dir = Path(part_dir)
    index = part_dir / "clutters.json"
    seed_by_name = {}
    if index.is_file():
        seed_by_name = {Path(e["report"]).name: int(e["seed"]) for e in json.loads(index.read_text())}
    seeds: dict[int, Path] = {}
    for report in sorted([*part_dir.glob("builds/*.json"), *part_dir.glob("restores/*.json")]):
        match = _SEED_IN_NAME.search(report.name)
        seed = int(match.group(1)) if match else seed_by_name.get(report.name)
        if seed is not None:
            seeds.setdefault(seed, report)
    return seeds
