# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The harness ladder's rungs as VLM agent adapters for RBY1-M Wuji Hand 2's joint embodiment."""

from __future__ import annotations

import math
from typing import Literal

import isaaclab.utils.math as math_utils
import numpy as np
import torch
from pydantic import BaseModel, create_model

from dexbench_arena.embodiments.joint_state import as_torch
from dexbench_arena.embodiments.rby1_wuji2.constants import (
    RBY1_TORSO_LINK_NAMES,
    RBY1_WUJI2_HAND_JOINT_ACTION_NAMES,
    RBY1_WUJI2_HEAD_JOINT_NAMES,
    RBY1_WUJI2_LEFT_ARM_JOINT_NAMES,
    RBY1_WUJI2_RIGHT_ARM_JOINT_NAMES,
    RBY1_WUJI2_TORSO_JOINT_NAMES,
)
from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import (
    DexBenchRby1Wuji2JointEmbodiment,
    _rby1_wuji2_pink_kinematics,
)
from dexbench_arena.policy.hold_pose_policy import joint_position_action
from dexbench_arena_vlm_agent.upstream.action_adapter import GoalActionAdapter, InvalidAgentResponse
from dexbench_arena_vlm_agent.upstream.observation_adapter import VLMObservationAdapter

HANDS = ("left", "right")
BODY_JOINT_NAMES = (*RBY1_WUJI2_TORSO_JOINT_NAMES, *RBY1_WUJI2_HEAD_JOINT_NAMES)
JOINT_NAMES = (*RBY1_WUJI2_LEFT_ARM_JOINT_NAMES, *RBY1_WUJI2_RIGHT_ARM_JOINT_NAMES, *BODY_JOINT_NAMES,
               *RBY1_WUJI2_HAND_JOINT_ACTION_NAMES)
"""Every joint of the 62-D action: what L0 commands."""
JOINT_ACTION_TERM = "joint_pos"
CAMERAS = ("robot_pov_left_cam_rgb", "robot_pov_right_cam_rgb")

HAND_SHAPES = {
    "claw": {
        "thumb_cmc_flex": 1.292, "index_finger_mcp_flex": 0.043, "middle_finger_mcp_flex": 0.069,
        "ring_finger_mcp_flex": 0.044, "pinky_mcp_flex": 0.161, "thumb_cmc_abd": -0.007, "index_finger_mcp_abd": 0.235,
        "middle_finger_mcp_abd": 0.082, "ring_finger_mcp_abd": -0.385, "pinky_mcp_abd": -0.610, "thumb_mcp": 0.314,
        "index_finger_pip": 1.047, "middle_finger_pip": 1.081, "ring_finger_pip": 1.178, "pinky_pip": 0.627,
        "thumb_ip": 0.000, "index_finger_dip": 0.789, "middle_finger_dip": 0.745, "ring_finger_dip": 0.816,
        "pinky_dip": 1.228,
    },
    "cup": {
        "thumb_cmc_flex": 0.650, "index_finger_mcp_flex": 0.159, "middle_finger_mcp_flex": 0.159,
        "ring_finger_mcp_flex": 0.159, "pinky_mcp_flex": 0.159, "thumb_cmc_abd": 0.120, "index_finger_mcp_abd": 0.299,
        "middle_finger_mcp_abd": 0.179, "ring_finger_mcp_abd": 0.030, "pinky_mcp_abd": -0.080, "thumb_mcp": 0.020,
        "index_finger_pip": 0.090, "middle_finger_pip": 0.090, "ring_finger_pip": 0.090, "pinky_pip": 0.090,
        "thumb_ip": 0.150, "index_finger_dip": 0.200, "middle_finger_dip": 0.200, "ring_finger_dip": 0.200,
        "pinky_dip": 0.200,
    },
    "flat": {
        "thumb_cmc_flex": 0.327, "index_finger_mcp_flex": 0.204, "middle_finger_mcp_flex": 0.137,
        "ring_finger_mcp_flex": 0.045, "pinky_mcp_flex": 0.133, "thumb_cmc_abd": 0.527, "index_finger_mcp_abd": 0.338,
        "middle_finger_mcp_abd": 0.385, "ring_finger_mcp_abd": 0.330, "pinky_mcp_abd": 0.407, "thumb_mcp": -0.076,
        "index_finger_pip": -0.022, "middle_finger_pip": -0.022, "ring_finger_pip": -0.023, "pinky_pip": -0.021,
        "thumb_ip": 0.002, "index_finger_dip": -0.000, "middle_finger_dip": -0.000, "ring_finger_dip": -0.000,
        "pinky_dip": -0.001,
    },
    "thumb_tuck": {
        "thumb_cmc_flex": -1.111, "index_finger_mcp_flex": 0.000, "middle_finger_mcp_flex": 0.000,
        "ring_finger_mcp_flex": 0.000, "pinky_mcp_flex": 0.000, "thumb_cmc_abd": -1.484, "index_finger_mcp_abd": 0.000,
        "middle_finger_mcp_abd": 0.000, "ring_finger_mcp_abd": 0.000, "pinky_mcp_abd": 0.000, "thumb_mcp": -0.300,
        "index_finger_pip": 0.000, "middle_finger_pip": 0.000, "ring_finger_pip": 0.000, "pinky_pip": 0.000,
        "thumb_ip": 0.000, "index_finger_dip": 0.000, "middle_finger_dip": 0.000, "ring_finger_dip": 0.000,
        "pinky_dip": 0.000,
    },
    "curl": {
        "thumb_cmc_flex": 1.287, "index_finger_mcp_flex": 0.804, "middle_finger_mcp_flex": 0.809,
        "ring_finger_mcp_flex": 1.233, "pinky_mcp_flex": 1.233, "thumb_cmc_abd": -1.056, "index_finger_mcp_abd": 0.000,
        "middle_finger_mcp_abd": 0.000, "ring_finger_mcp_abd": 0.000, "pinky_mcp_abd": 0.000, "thumb_mcp": -0.160,
        "index_finger_pip": 0.487, "middle_finger_pip": 0.497, "ring_finger_pip": 1.233, "pinky_pip": 1.233,
        "thumb_ip": 0.264, "index_finger_dip": 0.581, "middle_finger_dip": 0.588, "ring_finger_dip": 0.986,
        "pinky_dip": 0.986,
    },
}
EIGENGRASP = {
    "thumb_cmc_flex": 1.404, "index_finger_mcp_flex": 0.395, "middle_finger_mcp_flex": 0.420,
    "ring_finger_mcp_flex": 0.581, "pinky_mcp_flex": 0.646, "thumb_cmc_abd": -0.252, "index_finger_mcp_abd": 0.280,
    "middle_finger_mcp_abd": 0.100, "ring_finger_mcp_abd": -0.184, "pinky_mcp_abd": -0.297, "thumb_mcp": 0.175,
    "index_finger_pip": 0.760, "middle_finger_pip": 0.763, "ring_finger_pip": 1.122, "pinky_pip": 0.935,
    "thumb_ip": 0.161, "index_finger_dip": 0.727, "middle_finger_dip": 0.675, "ring_finger_dip": 0.911,
    "pinky_dip": 1.134,
}
ZERO_ROTATION = {
    "left": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    "right": np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
}
GRASP_POINT = {"left": np.array([0.122, -0.018, -0.027]), "right": np.array([0.122, 0.018, -0.027])}
WORKSPACE = (np.array([-0.10, -0.80, 0.55]), np.array([0.95, 0.80, 1.60]))
"""Base-frame box (min, max) a grasp point target must lie in."""
TORSO_WORKSPACE = (np.array([-0.10, -0.95, -0.65]), np.array([0.95, 0.95, 0.50]))
"""Torso-frame box (min, max) a grasp point target must lie in. It holds every grasp point of the four EA tasks' demos
with at least 6 cm to spare: all frames of each task's EA LeRobot root (``gr00t_62d_320x180``, state and action, through
the Pink model) span x -0.02 to 0.80, y -0.88 (L2) to 0.86 (M1), z -0.49 to 0.42 m. The torso frame carries the arms, so
the box does not depend on the torso joints; the arms reach about 1.03 m to either side."""
WORKSPACES = {"base": WORKSPACE, "torso": TORSO_WORKSPACE}
REFERENCE_FRAMES = ("base", "torso")
"""What the hands the model reads and commands are in: the robot base frame, or the torso frame."""
TORSO_FRAME_LINK = RBY1_TORSO_LINK_NAMES[-1]
"""The torso frame: the link both arms hang from (their first joints' parent)."""
EEF_CONTROLLERS = ("eef_abs", "eef_delta")
"""What a hand pose goal is: absolute, or a change from where the hand is measured when the goal arrives."""
ANGLE_RANGES_DEG = {"pitch_deg": (-180.0, 180.0), "roll_deg": (-90.0, 90.0), "yaw_deg": (-180.0, 180.0)}
EEF_AXES = ("x", "y", "z", "pitch_deg", "roll_deg", "yaw_deg")
DELTA_AXES = ("dx", "dy", "dz", "dpitch_deg", "droll_deg", "dyaw_deg")
"""An ``eef_delta`` goal's pose fields per hand, in :data:`EEF_AXES` order: the grasp point's change (m) and the turn
about the frame's axes (deg). The hand's grasp fields stay absolute."""
CHANGE_RANGES = {"dx": (-0.5, 0.5), "dy": (-0.5, 0.5), "dz": (-0.5, 0.5), "dpitch_deg": (-180.0, 180.0),
                 "droll_deg": (-90.0, 90.0), "dyaw_deg": (-180.0, 180.0)}
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
_FINGER_PREFIX = {"thumb": "thumb_", "index": "index_finger_", "middle": "middle_finger_", "ring": "ring_finger_",
                  "pinky": "pinky_"}

COMMON_PROMPT = """You are controlling a robot embodiment, a Rainbow RB-Y1 humanoid with two 20-joint Wuji dexterous hands, in a physics simulator. Each turn you receive the RGB images of its two head cameras (a stereo pair), the robot state and the task. Answer with a goal: set only the fields you want to change and leave every other one null, where it is held. The robot moves toward the goal at bounded speed and you are asked again once it arrives, or after {goal_s:.0f} s if it cannot. Write what you see and why you chose this goal in `note`. {frame} There is no depth image, object pose or camera calibration: estimate positions from the stereo pair and from where the hands appear. The episode ends by itself when the task is judged complete or failed, or when its time runs out; the state gives the time left.
{recipe}"""

JOINT_PROMPT = COMMON_PROMPT + """
Your goal is absolute joint targets in radians, one field per joint; a target past its limits is clamped. Limits: {bounds}. torso_0 and torso_4 tilt the upper body sideways, torso_1 to torso_3 fold it forward and back, torso_5 turns it (+ left); head_0 turns the head (+ looks left), head_1 tips it (+ looks down); the cameras ride on the head. Each arm has seven joints, shoulder (arm_0) to wrist (arm_6); each hand twenty, four per finger: knuckle bend (thumb_cmc_flex, *_mcp_flex), sideways spread (thumb_cmc_abd, *_mcp_abd), middle joint (thumb_mcp, *_pip) and end joint (thumb_ip, *_dip). Every joint moves at up to {joint_speed:.2f} rad/s."""

EEF_PROMPT = COMMON_PROMPT + """
{pose_goal} its hand: {hand}; and the torso and neck joints in radians (torso_0 and torso_4 tilt sideways, torso_1 to torso_3 fold forward and back, torso_5 turns, + left; head_0 turns the head, + looks left; head_1 tips it, + looks down; the cameras ride on the head). Bounds, outside which a goal is refused: {bounds}. Hand geometry at orientation (0, 0, 0), from the grasp point: the wrist is 0.12 m behind and 0.03 m above it, the palm (9 cm wide) about 0.02 m above it; open, the fingertips reach about 2.5 cm past it. A hand keeps its start shape until you first set its grasp. The grasp point moves in a straight line at {speed:.2f} m/s and turns at up to {turn_speed:.0f} deg/s, a hand closes over {grasp_s:.1f} s, a body joint moves at up to {joint_speed:.2f} rad/s; the arms reach about 0.69 m from the shoulders. Lift clear before travelling over other objects; within 2 cm of contact move 1 cm at a time; after closing, check in the images whether the object came with the hand before carrying it."""
HAND_GENERIC = "its closure `<hand>_grasp` from 0 (open) to 1 (closed) along one fixed grasp, and `<hand>_grasp_<finger>` (thumb, index, middle, ring, pinky) to close one finger more or less than the rest"
HAND_SHAPES_TEXT = ("its shape `<hand>_grasp_shape`, one of {shapes}, its closure `<hand>_grasp` from 0 (open) to 1 (the full shape), "
                    "and `<hand>_grasp_<finger>` (thumb, index, middle, ring, pinky) to close one finger more or less than the "
                    "rest. The shapes at closure 1: claw, thumb opposed and fingers curled at the middle and end joints; cup, "
                    "a light curl everywhere, a supporting palm; flat, a nearly open hand with the fingers spread; "
                    "thumb_tuck, fingers straight and the thumb folded under them; curl, ring and pinky curled, index and "
                    "middle half curled")
FRAME_TEXT = {
    "base": ("Positions are in the robot base frame, metres: +x forward, +y to the robot's left, +z up, origin at the "
             "base on the floor; the base does not move."),
    "torso": (
        "Hand positions and orientations are in the torso frame, metres: origin at the upper torso (link_torso_5, the "
        "link both arms hang from), +x forward out of the chest, +y to the robot's left, +z up along the spine. It turns "
        "and bends with the torso, so the hands go with it; with the torso joints at zero it is the base frame raised "
        "about 1.1 m. The state gives its pose in the robot base frame (+x forward, +y to the robot's left, +z up, origin "
        "at the base on the floor; the base does not move), the frame any fixed height in the task is given in."
    ),
}
"""The prompts' ``{frame}``, per :data:`REFERENCE_FRAMES`: what positions are given in. L0 takes the base one."""
FRAME_AXES_TEXT = {"base": "the fixed base axes", "torso": "the torso frame's axes"}
POSE_GOAL_TEXT = {
    "eef_abs": (
        "Your goal is, per hand, its grasp point `<hand>_x/_y/_z` (the centre of the five fingertip pads of the closed "
        "hand: name the centre of what you want to grasp, there is no offset to add) and its orientation "
        "`<hand>_pitch_deg/_roll_deg/_yaw_deg` in degrees from the straight-down hand (fingers forward +x, palm down "
        "-z), applied about {axes} pitch (y) first, then roll (x), then yaw (z) (+90 pitch points the fingers down, +90 "
        "roll turns the palm to the left, +90 yaw points the fingers to the left);"
    ),
    "eef_delta": (
        "Your goal is, per hand, a change from where the hand is when your goal arrives (measured then, not where you "
        "last aimed it): `<hand>_dx/_dy/_dz` moves its grasp point (the centre of the five fingertip pads of the closed "
        "hand) by that many metres along the frame's axes, and `<hand>_dpitch_deg/_droll_deg/_dyaw_deg` turn the hand "
        "by that many degrees about the frame's fixed axes, pitch (y) first, then roll (x), then yaw (z), the grasp "
        "point staying put (+ pitch tips fingers that point forward down, + roll turns a palm that faces down to the "
        "left, + yaw turns fingers that point forward to the left); the state gives where each grasp point is and how "
        "each hand is turned now; the grasp fields are absolute targets, not changes;"
    ),
}
"""L1's ``{pose_goal}``, per :data:`EEF_CONTROLLERS`: what a hand pose goal is; ``{axes}`` is
:data:`FRAME_AXES_TEXT`'s."""


def _optional_fields(names) -> type[BaseModel]:
    return create_model("Goal", note=(str, ...), **{name: (float | None, None) for name in names})


def _sets_nothing(response) -> bool:
    """Whether a goal names no field (only its note): the model chose to wait. Such a goal is where the robot already
    is, so it would count as reached at once and the policy would ask again on the next step, a burst of calls; it
    is held for ``max_steps`` instead, like a hold."""
    return all(value is None for name, value in response if name != "note")


def _fmt(value: float) -> str:
    return f"{float(value):.4g}"


class _RobotGoalAdapter(GoalActionAdapter):
    """What both rungs share: the env they bind to on the first step, the joint action term, the time left."""

    def __init__(self, max_steps: int):
        super().__init__(max_steps)
        self._env = None
        self._last: np.ndarray | None = None
        self.holding = False

    def is_robot_action_space_supported(self, env) -> bool:
        unwrapped = env.unwrapped
        manager = unwrapped.action_manager
        names = list(unwrapped.scene["robot"].data.joint_names)
        term = manager.get_term(JOINT_ACTION_TERM) if list(manager.active_terms) == [JOINT_ACTION_TERM] else None
        covered = [] if term is None else [names[i] for i in (range(len(names)) if isinstance(term._joint_ids, slice)
                                                              else term._joint_ids)]
        supported = unwrapped.num_envs == 1 and sorted(covered) == sorted(JOINT_NAMES)
        if supported and self._env is None:
            self._env, self._term, self._names = unwrapped, term, names
            self._ids = [names.index(name) for name in JOINT_NAMES]
            self._limits = as_torch(unwrapped.scene["robot"].data.soft_joint_pos_limits)[0].cpu().numpy().astype(np.float64)
            self._bind()
        return supported

    def _bind(self) -> None:
        pass

    def reset(self) -> None:
        self._last, self.holding = None, False

    def hold_goal(self, proprioception: dict) -> np.ndarray:
        """A goal of the targets as last commanded (L1: unnamed axes keep the last goal; L0: the joints as measured),
        held until ``max_steps`` pass: never reached early, so the policy does not ask again on the next step."""
        goal = self.decode_command(self.response_model(note="hold"), proprioception)
        self.holding = True
        return goal

    def _joint_pos(self) -> np.ndarray:
        return as_torch(self._env.scene["robot"].data.joint_pos)[0].detach().cpu().numpy().astype(np.float64)

    def _joint_target(self) -> np.ndarray:
        return as_torch(self._env.scene["robot"].data.joint_pos_target)[0].detach().cpu().numpy().astype(np.float64)

    def _time_left_s(self) -> float:
        env = self._env
        return round(float(env.max_episode_length - int(env.episode_length_buf[0])) * float(env.step_dt), 1)

    def _action(self, joint_targets: dict[int, float] | np.ndarray, ids=None) -> torch.Tensor:
        full = as_torch(self._env.scene["robot"].data.joint_pos).detach().clone()
        full[0, ids if ids is not None else self._ids] = torch.as_tensor(joint_targets, dtype=full.dtype, device=full.device)
        return joint_position_action(self._term, full)[0]


class JointGoalAdapter(_RobotGoalAdapter):
    """L0: the 62 absolute joint targets."""

    response_model = _optional_fields(JOINT_NAMES)

    def __init__(self, max_steps: int, joint_speed: float, goal_tolerance_rad: float = 0.02):
        super().__init__(max_steps)
        self.joint_speed, self.tolerance = joint_speed, goal_tolerance_rad

    def system_prompt(self, recipe: str, goal_s: float) -> str:
        bounds = ", ".join(f"{name} [{_fmt(lo)}, {_fmt(hi)}]" for name, (lo, hi) in zip(JOINT_NAMES, self._limits[self._ids]))
        return JOINT_PROMPT.format(recipe=recipe, goal_s=goal_s, frame=FRAME_TEXT["base"], bounds=bounds,
                                   joint_speed=self.joint_speed)

    def state(self) -> dict:
        positions = self._joint_pos()[self._ids]
        return {"time_left_s": self._time_left_s(),
                "joint_positions_rad": {name: round(float(value), 4) for name, value in zip(JOINT_NAMES, positions)}}

    def decode_command(self, response, proprioception: dict) -> np.ndarray:
        self.holding = _sets_nothing(response)
        goal = np.array([proprioception["joint_positions_rad"][name] for name in JOINT_NAMES])
        for index, name in enumerate(JOINT_NAMES):
            value = getattr(response, name)
            if value is not None:
                goal[index] = value
        return np.clip(goal, *self._limits[self._ids].T)

    def goal_reached(self, goal: np.ndarray, proprioception: dict) -> bool:
        if self.holding:
            return False
        positions = np.array([proprioception["joint_positions_rad"][name] for name in JOINT_NAMES])
        return float(np.max(np.abs(goal - positions))) < self.tolerance

    def compute_command(self, goal: np.ndarray, proprioception: dict) -> np.ndarray:
        last = self._joint_target()[self._ids] if self._last is None else self._last
        step = self.joint_speed * float(self._env.step_dt)
        self._last = last + np.clip(goal - last, -step, step)
        return self._last

    def command_to_action(self, env, command: np.ndarray) -> torch.Tensor:
        return self._action(command).to(env.unwrapped.device)


class EefGoalAdapter(_RobotGoalAdapter):
    """L1: grasp points and orientations through arm IK, a hand closure per finger along a shape, torso and neck joints."""

    def __init__(self, max_steps: int, hand: str, ik_profile: str, linear_speed: float, angular_speed_deg: float,
                 joint_speed: float, grasp_s: float, reference_frame: str = "base", controller: str = "eef_abs"):
        super().__init__(max_steps)
        assert hand in ("generic", "shapes"), f"hand {hand!r}: generic or shapes"
        assert reference_frame in REFERENCE_FRAMES, f"reference_frame {reference_frame!r}: one of {REFERENCE_FRAMES}"
        assert controller in EEF_CONTROLLERS, f"controller {controller!r}: one of {EEF_CONTROLLERS}"
        self.hand, self.reference_frame, self.goal_kind = hand, reference_frame, controller
        self.shapes = dict(HAND_SHAPES) if hand == "shapes" else {"eigengrasp": EIGENGRASP}
        pose_axes = DELTA_AXES if controller == "eef_delta" else EEF_AXES
        fields = {f"{side}_{axis}": (float | None, None) for side in HANDS for axis in (*pose_axes, "grasp",
                                                                                       *(f"grasp_{f}" for f in FINGERS))}
        if hand == "shapes":
            fields.update({f"{side}_grasp_shape": (Literal[tuple(self.shapes)] | None, None) for side in HANDS})
        fields.update({name: (float | None, None) for name in BODY_JOINT_NAMES})
        self.response_model = create_model("Goal", note=(str, ...), **fields)
        self.ik_profile, self.linear_speed, self.angular_speed_deg = ik_profile, linear_speed, angular_speed_deg
        self.joint_speed, self.grasp_s = joint_speed, grasp_s
        self.controller = None

    def _bind(self) -> None:
        self.controller = DexBenchRby1Wuji2JointEmbodiment.eef_arm_ik_controller(self._env, self.ik_profile)
        self._kinematics = _rby1_wuji2_pink_kinematics()
        self._pink_order = [self._names.index(name) for name in self._kinematics._all_joint_names]
        self._hand_ids = {side: [self._names.index(n) for n in RBY1_WUJI2_HAND_JOINT_ACTION_NAMES if n.startswith(f"{side}_")]
                          for side in HANDS}
        self._suffixes = [n.split("_", 1)[1] for n in RBY1_WUJI2_HAND_JOINT_ACTION_NAMES if n.startswith("left_")]
        self._finger = {f: np.array([s.startswith(_FINGER_PREFIX[f]) for s in self._suffixes]) for f in FINGERS}
        self._shape_vec = {name: np.array([shape[s] for s in self._suffixes]) for name, shape in self.shapes.items()}
        self._body_ids = [self._names.index(name) for name in BODY_JOINT_NAMES]
        zero = {side: math_utils.quat_from_matrix(torch.as_tensor(ZERO_ROTATION[side]))[None] for side in HANDS}
        self._offset = {side: math_utils.quat_apply_inverse(zero[side], torch.as_tensor(GRASP_POINT[side])[None])
                        for side in HANDS}
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._shape = {side: next(iter(self.shapes)) for side in HANDS}
        self._hand_goal: dict[str, np.ndarray] = {}
        self._pose_goal: dict[str, list[float]] = {}
        self._body_goal: np.ndarray | None = None
        if self.controller is not None:
            self.controller.reset(None)

    def system_prompt(self, recipe: str, goal_s: float) -> str:
        lo, hi = WORKSPACES[self.reference_frame]
        box = [f"<hand>_{axis} [{_fmt(lo[i])}, {_fmt(hi[i])}]" for i, axis in enumerate("xyz")]
        if self.goal_kind == "eef_delta":
            bounds = [f"<hand>_{axis} [{_fmt(a)}, {_fmt(b)}]" for axis, (a, b) in CHANGE_RANGES.items()]
            bounds.append("a grasp point must end within " + ", ".join(box))
        else:
            bounds = box + [f"<hand>_{axis} [{_fmt(a)}, {_fmt(b)}]" for axis, (a, b) in ANGLE_RANGES_DEG.items()]
        bounds += ["<hand>_grasp and <hand>_grasp_<finger> [0, 1]"]
        bounds += [f"{name} [{_fmt(a)}, {_fmt(b)}]" for name, (a, b) in zip(BODY_JOINT_NAMES, self._limits[self._body_ids])]
        hand = HAND_GENERIC if self.hand == "generic" else HAND_SHAPES_TEXT.format(shapes=", ".join(self.shapes))
        pose_goal = POSE_GOAL_TEXT[self.goal_kind].format(axes=FRAME_AXES_TEXT[self.reference_frame])
        return EEF_PROMPT.format(recipe=recipe, goal_s=goal_s, frame=FRAME_TEXT[self.reference_frame], pose_goal=pose_goal,
                                 bounds=", ".join(bounds), hand=hand, speed=self.linear_speed,
                                 turn_speed=self.angular_speed_deg, grasp_s=self.grasp_s, joint_speed=self.joint_speed)

    def _frame(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The reference frame's pose in the base frame at the kinematics' current joints (the torso link's); None for
        the base frame itself."""
        if self.reference_frame == "base":
            return None
        return _pose(self._kinematics.get_transform_frame_to_world(TORSO_FRAME_LINK))

    def _grasp_poses(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Each hand's grasp point and orientation (x, y, z, w) as measured, in the reference frame."""
        self._kinematics.update(self._joint_pos()[self._pink_order])
        frame, poses = self._frame(), {}
        for side in HANDS:
            wrist = _pose(self._kinematics.get_transform_frame_to_world(f"{side}_wrist_frame"))
            pose = math_utils.combine_frame_transforms(*wrist, self._offset[side])
            poses[side] = pose if frame is None else math_utils.subtract_frame_transforms(*frame, *pose)
        return poses

    def _closures(self, joints: np.ndarray, shape: str) -> np.ndarray:
        """Per finger, how far ``joints`` are along ``shape``, from 0 (open) to 1."""
        vec, out = self._shape_vec[shape], np.zeros(len(FINGERS))
        for i, finger in enumerate(FINGERS):
            moving = self._finger[finger] & (np.abs(vec) > 0.05)
            if moving.any():
                out[i] = float(np.clip(np.mean(joints[moving] / vec[moving]), 0.0, 1.0))
        return out

    def state(self) -> dict:
        state = {"time_left_s": self._time_left_s()}
        joint_pos = self._joint_pos()
        for side, (point, quat) in self._grasp_poses().items():
            angles = _euler_deg(side, math_utils.matrix_from_quat(quat)[0].numpy())
            state.update({f"{side}_{axis}": round(float(value), 4) for axis, value in zip("xyz", point[0])})
            state.update({f"{side}_{axis}": round(value, 1) for axis, value in zip(EEF_AXES[3:], angles)})
            closures = self._closures(joint_pos[self._hand_ids[side]], self._shape[side])
            if self.hand == "shapes":
                state[f"{side}_grasp_shape"] = self._shape[side]
            state[f"{side}_grasp"] = round(float(np.mean(closures)), 2)
            state.update({f"{side}_grasp_{f}": round(float(c), 2) for f, c in zip(FINGERS, closures)})
        state.update({name: round(float(v), 4) for name, v in zip(BODY_JOINT_NAMES, joint_pos[self._body_ids])})
        frame = self._frame()
        if frame is not None:
            origin, quat = frame
            roll, pitch, yaw = (round(math.degrees(float(angle[0])), 1) for angle in math_utils.euler_xyz_from_quat(quat))
            state["torso_frame_in_base"] = {"origin_m": [round(float(value), 4) for value in origin[0]],
                                            "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll}
        return state

    def decode_command(self, response, proprioception: dict) -> np.ndarray:
        """Per hand: grasp point (3), angles (3), hand joint targets (20); then the body joints (8). The pose is in the
        reference frame; an ``eef_delta`` one is where the change ends (:meth:`_changed_poses`)."""
        self.holding = _sets_nothing(response)
        lo, hi = WORKSPACES[self.reference_frame]
        if self.goal_kind == "eef_delta":
            bounds = {f"{side}_{axis}": rng for side in HANDS for axis, rng in CHANGE_RANGES.items()}
        else:
            bounds = {f"{side}_{axis}": (lo[i], hi[i]) for side in HANDS for i, axis in enumerate("xyz")}
            bounds.update({f"{side}_{axis}": rng for side in HANDS for axis, rng in ANGLE_RANGES_DEG.items()})
        bounds.update({f"{side}_grasp{suffix}": (0.0, 1.0) for side in HANDS for suffix in ("", *(f"_{f}" for f in FINGERS))})
        bounds.update(dict(zip(BODY_JOINT_NAMES, map(tuple, self._limits[self._body_ids]))))
        for name, (low, high) in bounds.items():
            value = getattr(response, name)
            if value is not None and not low <= value <= high:
                raise InvalidAgentResponse(f"{name}={value} is outside its bounds [{_fmt(low)}, {_fmt(high)}]")
        changed = self._changed_poses(response) if self.goal_kind == "eef_delta" else None
        goal, poses = [], {}
        targets = self._joint_target()
        for side in HANDS:
            last = self._pose_goal.get(side) or [proprioception[f"{side}_{axis}"] for axis in EEF_AXES]
            poses[side] = changed[side] if changed is not None else [
                getattr(response, f"{side}_{axis}") if getattr(response, f"{side}_{axis}") is not None else last[i]
                for i, axis in enumerate(EEF_AXES)]
            goal += poses[side]
            named = {f: getattr(response, f"{side}_grasp_{f}") for f in FINGERS}
            whole = getattr(response, f"{side}_grasp")
            shape = getattr(response, f"{side}_grasp_shape", None)
            if whole is None and shape is None and all(v is None for v in named.values()):
                hand = self._hand_goal.get(side, targets[self._hand_ids[side]])
            else:
                shape = shape or self._shape[side]
                closures = (np.full(len(FINGERS), whole) if whole is not None else
                            self._closures(self._hand_goal.get(side, targets[self._hand_ids[side]]), shape))
                closures = np.array([named[f] if named[f] is not None else closures[i] for i, f in enumerate(FINGERS)])
                hand = np.zeros(len(self._suffixes))
                for i, finger in enumerate(FINGERS):
                    hand[self._finger[finger]] = closures[i] * self._shape_vec[shape][self._finger[finger]]
                self._shape[side], self._hand_goal[side] = shape, hand
            goal += list(hand)
        body_last = self._body_goal if self._body_goal is not None else targets[self._body_ids]
        body = [getattr(response, n) if getattr(response, n) is not None else float(body_last[i])
                for i, n in enumerate(BODY_JOINT_NAMES)]
        goal += body
        if changed is None:
            self._pose_goal = poses
        self._body_goal = np.array(body)
        return np.array(goal, dtype=np.float64)

    def _changed_poses(self, response) -> dict[str, list[float]]:
        """Per hand, the pose an ``eef_delta`` goal asks for, from the hand measured now: the grasp point moved by the
        change, the hand turned about the frame's axes (``Rz Rx Ry`` before its orientation, the grasp point kept). A
        change that would end the grasp point outside the workspace along its axis is refused; an axis not named stays
        where it is measured, unchecked, as an absolute goal's does (so a hold never fails)."""
        lo, hi = WORKSPACES[self.reference_frame]
        poses = {}
        for side, (point, quat) in self._grasp_poses().items():
            named = {axis: getattr(response, f"{side}_{axis}") for axis in DELTA_AXES}
            value = {axis: 0.0 if change is None else change for axis, change in named.items()}
            point = point[0].numpy() + np.array([value["dx"], value["dy"], value["dz"]])
            for i, axis in enumerate(("dx", "dy", "dz")):
                if named[axis] is not None and not lo[i] <= point[i] <= hi[i]:
                    raise InvalidAgentResponse(f"{side}_{axis}={named[axis]} would end the grasp point at {axis[1]}="
                                               f"{point[i]:.3f}, outside [{_fmt(lo[i])}, {_fmt(hi[i])}]")
            turn = _matrix_from_turn(value["dpitch_deg"], value["droll_deg"], value["dyaw_deg"])
            turned = math_utils.quat_mul(math_utils.quat_from_matrix(torch.as_tensor(turn))[None], quat)
            poses[side] = [*point, *_euler_deg(side, math_utils.matrix_from_quat(turned)[0].numpy())]
        return poses

    def goal_reached(self, goal: np.ndarray, proprioception: dict) -> bool:
        if self.holding:
            return False
        joint_pos, per_hand = self._joint_pos(), 6 + len(self._suffixes)
        for index, side in enumerate(HANDS):
            g = index * per_hand
            point = np.array([proprioception[f"{side}_{axis}"] for axis in "xyz"])
            if (np.linalg.norm(goal[g:g + 3] - point) > 0.01
                    or np.max(np.abs(goal[g + 6:g + per_hand] - joint_pos[self._hand_ids[side]])) > 0.1):
                return False
        body = np.array([proprioception[name] for name in BODY_JOINT_NAMES])
        return float(np.max(np.abs(goal[-len(BODY_JOINT_NAMES):] - body))) < 0.02

    def compute_command(self, goal: np.ndarray, proprioception: dict) -> np.ndarray:
        """One step toward the goal from the last commanded one: per hand the grasp point (3), its rotation as a
        quaternion (x, y, z, w) and its 20 hand joint targets; then the body joints."""
        dt, per_hand, per_cmd = float(self._env.step_dt), 6 + len(self._suffixes), 7 + len(self._suffixes)
        if self._last is None:
            last, targets = [], self._joint_target()
            for side, (point, quat) in self._grasp_poses().items():
                last += [*point[0].tolist(), *quat[0].tolist(), *targets[self._hand_ids[side]]]
            self._last = np.array([*last, *targets[self._body_ids]])
        command = self._last.copy()
        hand_step = dt * max(np.max(np.abs(v)) for v in self._shape_vec.values()) / self.grasp_s
        for index, side in enumerate(HANDS):
            g, c = index * per_hand, index * per_cmd
            delta = goal[g:g + 3] - command[c:c + 3]
            distance = float(np.linalg.norm(delta))
            if distance > 0:
                command[c:c + 3] += delta * min(1.0, self.linear_speed * dt / distance)
            target = math_utils.quat_from_matrix(torch.as_tensor(_matrix(side, *goal[g + 3:g + 6])))
            current = torch.as_tensor(command[c + 3:c + 7])
            angle = math.degrees(float(math_utils.quat_error_magnitude(current, target)))
            fraction = min(1.0, self.angular_speed_deg * dt / angle) if angle > 1e-6 else 1.0
            command[c + 3:c + 7] = math_utils.quat_slerp(current.clone(), target, fraction).numpy()
            hand = command[c + 7:c + per_cmd]
            command[c + 7:c + per_cmd] = hand + np.clip(goal[g + 6:g + per_hand] - hand, -hand_step, hand_step)
        body = command[2 * per_cmd:]
        command[2 * per_cmd:] = body + np.clip(goal[-len(BODY_JOINT_NAMES):] - body, -self.joint_speed * dt, self.joint_speed * dt)
        self._last = command
        return command

    def command_to_action(self, env, command: np.ndarray) -> torch.Tensor:
        per_cmd, wrists, hands = 7 + len(self._suffixes), {}, []
        if self.reference_frame != "base":
            self._kinematics.update(self._joint_pos()[self._pink_order])
        frame = self._frame()
        for index, side in enumerate(HANDS):
            c = index * per_cmd
            pose = (torch.as_tensor(command[c:c + 3], dtype=torch.float64)[None],
                    math_utils.normalize(torch.as_tensor(command[c + 3:c + 7], dtype=torch.float64))[None])
            if frame is not None:
                pose = math_utils.combine_frame_transforms(*frame, *pose)
            position, quat = math_utils.combine_frame_transforms(*pose, -self._offset[side])
            wrists[side] = (position.float(), math_utils.matrix_from_quat(quat).float())
            hands.append(command[c + 7:c + per_cmd])
        passed = torch.as_tensor(np.concatenate([*hands, command[2 * per_cmd:]]), dtype=torch.float32)[None]
        data = self._env.scene["robot"].data
        return self.controller.action(as_torch(data.joint_pos), as_torch(data.joint_pos_target), wrists, passed)[0].to(
            env.unwrapped.device)


def _pose(transform) -> tuple[torch.Tensor, torch.Tensor]:
    """A Pinocchio frame transform as Isaac Lab's position (1, 3) and quaternion (x, y, z, w) (1, 4), float64."""
    position = torch.tensor(np.asarray(transform.translation, dtype=np.float64))[None]
    return position, math_utils.quat_from_matrix(torch.tensor(np.asarray(transform.rotation, dtype=np.float64)))[None]


def _matrix_from_turn(pitch_deg: float, roll_deg: float, yaw_deg: float) -> np.ndarray:
    """A turn about the frame's fixed axes, pitch (y) first, then roll (x), then yaw (z): ``Rz @ Rx @ Ry``."""
    angles = torch.tensor([math.radians(float(v)) for v in (yaw_deg, roll_deg, pitch_deg)], dtype=torch.float64)
    return math_utils.matrix_from_euler(angles, "ZXY").numpy()


def _matrix(hand: str, pitch_deg: float, roll_deg: float, yaw_deg: float) -> np.ndarray:
    """(pitch, roll, yaw) from the straight-down hand -> wrist rotation: ``Rz(yaw) @ Rx(roll) @ Ry(pitch) @ R0``."""
    return _matrix_from_turn(pitch_deg, roll_deg, yaw_deg) @ ZERO_ROTATION[hand]


def _euler_deg(hand: str, rotation: np.ndarray) -> tuple[float, float, float]:
    """The inverse of :func:`_matrix`, roll in [-90, 90]. Isaac Lab offers only the XYZ extraction, so this ZXY one is
    written out."""
    rel = np.asarray(rotation, dtype=np.float64) @ ZERO_ROTATION[hand].T
    roll = math.asin(max(-1.0, min(1.0, rel[2, 1])))
    if abs(math.cos(roll)) < 1e-6:
        return 0.0, math.degrees(roll), math.degrees(math.atan2(rel[1, 0], rel[0, 0]))
    return math.degrees(math.atan2(-rel[2, 0], rel[2, 2])), math.degrees(roll), math.degrees(math.atan2(-rel[0, 1], rel[1, 1]))


class RobotStateObservationAdapter(VLMObservationAdapter):
    """The two head cameras, and the state its rung's adapter reads (one env)."""

    camera_keys = CAMERAS

    def __init__(self, action_adapter: _RobotGoalAdapter):
        self._action_adapter = action_adapter

    def extract_proprioception(self, env, observation) -> list[dict]:
        return [self._action_adapter.state()]
