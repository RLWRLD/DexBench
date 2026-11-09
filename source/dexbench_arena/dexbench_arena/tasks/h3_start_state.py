"""Validate archived H3 scene states before restoring them."""

from __future__ import annotations

import math


def _finite_vector(value, length: int, label: str):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must be a list of {length} finite numbers")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value):
        raise ValueError(f"{label} must contain finite numbers")


def validate_grasp_state(state: dict, object_names: list[str]) -> dict:
    """Validate every body and joint before constructing assets or restoring a dump."""
    if not isinstance(state, dict) or state.get("format") not in ("dexbench_h3_state_v1", "dexbench_h3_grasp_v1"):
        raise ValueError("Expected a dexbench_h3_state_v1 YAML state dump")
    if state.get("frame") != "env" or state.get("quaternion_order") != "xyzw":
        raise ValueError("H3 state dumps require frame: env and quaternion_order: xyzw")
    if state.get("scene_asset") != "dexbench_hotel_fb_e4_260831":
        raise ValueError("H3 state dump must use the shared E4 banquet scene")
    objects = state.get("objects")
    if not isinstance(objects, dict) or set(objects) != set(object_names):
        raise ValueError("Grasp dump utensils do not match the scene object names")
    robot = state.get("robot")
    if not isinstance(robot, dict):
        raise ValueError("Grasp dump needs robot state")
    joints = robot.get("joint_positions")
    velocities = robot.get("joint_velocities")
    if (
        not isinstance(joints, dict)
        or not joints
        or not isinstance(velocities, dict)
        or joints.keys() != velocities.keys()
    ):
        raise ValueError("Grasp dump needs matching named joint positions and velocities")
    for name in joints:
        if not isinstance(name, str):
            raise ValueError("Robot joint names must be strings")
        _finite_vector([joints[name], velocities[name]], 2, f"Joint {name}")
    for name, body in [("robot", robot), *objects.items()]:
        if not isinstance(body, dict):
            raise ValueError(f"{name} needs a root_pose and root_velocity")
        _finite_vector(body.get("root_pose"), 7, f"{name}.root_pose (XYZ + XYZW)")
        _finite_vector(body.get("root_velocity"), 6, f"{name}.root_velocity (linear + angular)")
        if not math.isclose(sum(v * v for v in body["root_pose"][3:]), 1.0, abs_tol=1e-4):
            raise ValueError(f"{name} quaternion must be normalized XYZW")
    return state


def validate_grasp_start(start: dict, object_names: list[str]) -> dict:
    """Validate measured state and the separate position-command table."""
    state = validate_grasp_state(start["state"], object_names)
    commands = start.get("joint_targets")
    if not isinstance(commands, dict) or commands.keys() != state["robot"]["joint_positions"].keys():
        raise ValueError("H3 commands must name every saved robot joint")
    _finite_vector(list(commands.values()), len(commands), "H3 joint targets")
    return start

