# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Record the configured physics settings used for scene export and replay."""

import argparse
import json
import math
from numbers import Integral, Real
import warnings

from dexbench_arena.evaluation.evaluation_signature import NEWTON_SHAPE_OVERRIDES_ATTRIBUTE


def take_presets(argv):
    """Split ``--presets`` out of a replay tool's command line -> ``(["--presets", value] or [], the rest)``."""
    launcher = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    launcher.add_argument("--presets")
    options, rest = launcher.parse_known_args(list(argv))
    return (["--presets", options.presets] if options.presets is not None else []), rest


def recorded_fps(env_args):
    """Read the actual action/observation rate from the native HDF5 clock."""
    sim_args = env_args.get("sim_args")
    if not isinstance(sim_args, dict):
        return None
    try:
        dt = float(sim_args["dt"])
        decimation = int(sim_args["decimation"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(dt) or dt <= 0.0 or decimation <= 0:
        raise ValueError(
            f"recording dt and decimation must be positive, got {dt}, {decimation}"
        )
    rate = 1.0 / (dt * decimation)
    rounded = round(rate)
    if not math.isclose(rate, rounded, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"recording control rate {rate} is not an integer FPS")
    return int(rounded)


def apply_recorded_timing(cfg, metadata):
    """Restore only the recorded step sizes, leaving the requested physics settings intact."""
    if recorded_fps(metadata) is None:
        raise ValueError("Replay requires sim_args.dt and sim_args.decimation in the input HDF5")
    cfg.sim.dt = float(metadata["sim_args"]["dt"])
    cfg.decimation = int(metadata["sim_args"]["decimation"])


def _scalar(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    raise TypeError(f"Physics metadata requires a JSON scalar, got {type(value).__name__}")


def _value(value):
    if isinstance(value, dict):
        return {str(key): _value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    return _scalar(value)


def _fields(cfg, names):
    return {name: _value(getattr(cfg, name)) for name in names if hasattr(cfg, name)}


def physics_runtime_metadata(cfg):
    """Read the final environment config after CLI/Hydra overrides are applied."""
    sim, physics = cfg.sim, cfg.sim.physics
    dt, decimation = float(sim.dt), int(cfg.decimation)
    settings = {
        "schema_version": 1,
        "backend": type(physics).__name__,
        "dt": dt,
        "decimation": decimation,
        **_fields(sim, ("use_newton_actuators",)),
    }
    if hasattr(sim, "gravity"):
        settings["gravity"] = [_scalar(value) for value in sim.gravity]
    if hasattr(physics, "num_substeps"):
        settings.update(_fields(physics, (
            "num_substeps", "collision_decimation", "use_cuda_graph", "deterministic_mode",
        )))
        solver = physics.solver_cfg
        settings["solver"] = {
            "config_type": type(solver).__name__,
            **_fields(solver, (
                "solver_type", "solver", "integrator", "update_data_interval", "iterations",
                "ls_iterations", "impratio", "cone", "tolerance", "ccd_iterations", "nconmax", "njmax",
                "use_mujoco_cpu", "use_mujoco_contacts", "enable_multiccd", "disable_contacts", "disable_sensors",
                "default_actuator_gear",
            )),
        }
        settings["default_shape"] = _fields(physics.default_shape_cfg, ("margin", "gap", "ke", "kd", "kf", "mu"))
        overrides = getattr(cfg, NEWTON_SHAPE_OVERRIDES_ATTRIBUTE, None)
        if overrides:
            settings["shape_overrides"] = _value(overrides)
    runtime = {"physics_backend": settings["backend"], "control_dt": dt * decimation, "physics_config": settings}
    json.dumps(runtime, allow_nan=False)
    return runtime


_DRIVE_PROPERTIES = {
    "stiffness": "joint_stiffness",
    "damping": "joint_damping",
    "armature": "joint_armature",
    "friction": "joint_friction_coeff",
    "viscous_friction": "joint_viscous_friction_coeff",
    "effort_limit": "joint_effort_limits",
    "velocity_limit": "joint_vel_limits",
}
_ACTUATOR_FIELDS = ("stiffness", "damping", "armature", "friction", "effort_limit", "velocity_limit")


def _finite(values):
    return [float(value) if math.isfinite(value) else None for value in values]


def robot_drive_metadata(env, asset_name="robot"):
    """Each robot joint's drive settings as the simulation was given them, for env 0."""
    robot = env.scene[asset_name]
    data = robot.data
    joints = {"joint_names": list(robot.joint_names)}
    for name, prop in _DRIVE_PROPERTIES.items():
        joints[name] = _finite(getattr(data, prop).torch[0].tolist())
    actuators = {}
    for group, actuator in robot.actuators.items():
        actuators[group] = {"class": type(actuator).__name__, "joint_names": list(actuator.joint_names)}
        for field in _ACTUATOR_FIELDS:
            value = getattr(actuator, field, None)
            if hasattr(value, "tolist"):
                actuators[group][field] = _finite(value[0].tolist())
    drive = {"schema_version": 1, "env_index": 0, "joints": joints, "actuators": actuators}
    json.dumps(drive, allow_nan=False)
    return drive


def _leaves(value, prefix=""):
    for name, item in value.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(item, dict):
            yield from _leaves(item, key)
        else:
            yield key, item


def physics_differences(first, second):
    """The settings two physics snapshots disagree on, by dotted key: ``{key: (first, second)}``."""
    previous, current = dict(_leaves(first)), dict(_leaves(second))
    differences = {}
    for key in sorted(previous.keys() | current.keys()):
        before, after = previous.get(key, "<missing>"), current.get(key, "<missing>")
        equal = before == after
        if isinstance(before, float) and isinstance(after, float):
            equal = math.isclose(before, after, rel_tol=1e-9, abs_tol=1e-12)
        if not equal:
            differences[key] = (before, after)
    return differences


def compare_source_physics(source_metadata, current_runtime, *, context):
    """Warn and return a persisted comparison; permit explicit A/B and inputs without ``physics_config``."""
    if "dexbench_rerecord" in source_metadata:
        location = "dexbench_rerecord.runtime"
        source_runtime = source_metadata["dexbench_rerecord"].get("runtime") or {}
    else:
        location = "dexbench_recording"
        source_runtime = source_metadata.get(location, {})
    saved = source_runtime.get("physics_config")
    if saved is None:
        warnings.warn(
            f"{context}: source {location}.physics_config is missing; allowing the input, "
            "but physics-setting compatibility is unverified.", RuntimeWarning, stacklevel=2,
        )
        return {"status": "unverified", "source": location, "reason": "missing_physics_config"}
    if not isinstance(saved, dict):
        raise ValueError(f"Invalid source {location}.physics_config: expected an object")
    pairs = physics_differences(saved, current_runtime["physics_config"])
    saved_drive, current_drive = source_runtime.get("robot_drive"), current_runtime.get("robot_drive")
    if saved_drive is not None and current_drive is not None:
        pairs.update(physics_differences({"robot_drive": saved_drive}, {"robot_drive": current_drive}))
    differences = {key: {"source": before, "current": after} for key, (before, after) in pairs.items()}
    if differences:
        details = "; ".join(f"{key}: source={v['source']!r}, current={v['current']!r}" for key, v in differences.items())
        warnings.warn(f"{context}: physics settings differ from {location}: {details}", RuntimeWarning, stacklevel=2)
        return {"status": "mismatch", "source": location, "differences": differences}
    if saved_drive is None or current_drive is None:
        side = f"source {location}" if saved_drive is None else "this run"
        warnings.warn(
            f"{context}: {side} has no robot_drive, so the robot's joint gains are unverified.",
            RuntimeWarning, stacklevel=2,
        )
        return {"status": "unverified", "source": location, "reason": "missing_robot_drive", "differences": {}}
    return {"status": "match", "source": location, "differences": differences}
