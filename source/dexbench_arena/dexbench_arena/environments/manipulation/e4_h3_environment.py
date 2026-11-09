# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""H3 saved-bundle extraction runtime with digest-bound benchmark inputs."""

from dataclasses import asdict, dataclass, replace
import math
import re

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory

from dexbench_arena import ASSETS_DIR
from dexbench_arena.environments.manipulation.scene_environment_base import DexBenchEnvironmentCfg, DexBenchSceneEnvironmentBase
from dexbench_arena.benchmark.protocol import CONTROL_HZ
from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_JOINT_EMBODIMENT
from dexbench_arena.environments.manipulation.newton_shape_contact import ShapeContactRequest, install_shape_contact
from dexbench_arena.environments.manipulation.newton_rolling_friction import ROLLING_CONDIM, install_rolling_friction
from dexbench_arena.environments.manipulation.newton_solref import solref_to_newton_ke_kd
from dexbench_arena.environments.manipulation.newton_isotropic_inertia import install_isotropic_inertia
from dexbench_arena.evaluation.evaluation_signature import NEWTON_SHAPE_OVERRIDES_ATTRIBUTE
from dexbench_arena.tasks.h3_runtime_bundle import load_h3_runtime_bundle
from dexbench_arena.tasks.h3_objects import (
    H3_INPUTS_DIR, H3_OBJECT_AXIS, H3_OBJECTS, H3_PACKAGES, resolve_h3_bundle,
)


H3_UTENSIL_INERTIA_NAME = "h3_pr398_utensils"


@dataclass
class H3EnvironmentCfg(DexBenchEnvironmentCfg):
    task: str = 'extract_utensil'
    embodiment: str = RBY1_WUJI2_JOINT_EMBODIMENT
    anchor: str = 'A_h3'
    runtime_bundle: str = ''
    runtime_bundle_sha256: str = ''
    h3_object: str = ''
    h3_inputs_dir: str = H3_INPUTS_DIR
    rby1_control_hz: int = int(CONTROL_HZ)
    camera_scale: float = .5
    episode_length_s: float = 25.
    head_yaw_deg: float = 5.
    head_pitch_deg: float = 45.
    h3_drive_profile: str = 'collection_v1'


@register_environment
class H3Environment(DexBenchSceneEnvironmentBase, ArenaEnvironmentFactory[H3EnvironmentCfg]):
    name = 'dexbench_hotel_fb_e4_h3_environment'
    _legacy_argparse_cfg_type = H3EnvironmentCfg
    robot_base_anchors_xyz = {'A_h3': (3.55, 1., 0.)}
    anchor_facing_targets_xy = {'A_h3': (3.55, 0.)}
    default_anchor = 'A_h3'
    replay_hidden_prim_patterns = (r'^GroundPlane$',)
    object_axes = {'extract_utensil': H3_OBJECT_AXIS}

    def build(self, cfg):
        if cfg.task != 'extract_utensil' or cfg.start_pose or cfg.base_jitter_radius:
            raise ValueError('H3 uses a saved extraction state; start-pose/base jitter overrides are not supported')
        if cfg.arm_mode != 'dual_arm' or cfg.mimic:
            raise ValueError('H3 requires dual-arm joint control; Mimic integration is not yet provided')
        path, digest = cfg.runtime_bundle, cfg.runtime_bundle_sha256
        if not path:
            key = cfg.h3_object or H3_OBJECT_AXIS.default
            path, digest = resolve_h3_bundle(key, cfg.h3_inputs_dir)
        self.bundle = load_h3_runtime_bundle(path, digest)
        key = self.bundle['trio'] + '_' + self.bundle['target']
        obj = H3_OBJECT_AXIS.get(key)
        if self.bundle['cell'] != ('train' if obj.split == 'train' else 'ga_os'):
            raise ValueError('H3 runtime bundle cell does not match its object split')
        if cfg.h3_object and cfg.h3_object != key:
            raise ValueError('H3 object selection does not match the supplied runtime bundle')
        actual_assets = {a['type']: {'id': a['id'], 'sha256': a['sha256']} for a in self.bundle['assets']}
        if actual_assets != H3_OBJECTS[key]['assets']:
            raise ValueError('H3 bundle does not use the declared utensil assets')
        cfg = replace(cfg, h3_object=key)
        self.head_joint_positions = {'head_0': math.radians(cfg.head_yaw_deg),
                                     'head_1': math.radians(cfg.head_pitch_deg)}
        if not all(math.isfinite(v) for v in self.head_joint_positions.values()):
            raise ValueError('H3 head gaze must be finite')
        return super().build(cfg)

    def embodiment_constructor_kwargs(self, cfg):
        return dict(super().embodiment_constructor_kwargs(cfg), support_hand_mode='init_q',
                    extraction_drive_profile=cfg.h3_drive_profile)

    def _build_embodiment(self, cfg):
        from isaaclab_arena.utils.pose import Pose

        embodiment, teleop = super()._build_embodiment(cfg)
        action = embodiment.action_config.joint_pos
        if (asdict(action.grasp_profile) != self.bundle['support_hand_profile']
                or asdict(action.adaptive_profile) != self.bundle['adaptive_support_profile']):
            raise ValueError('Runtime support profile does not match the supplied H3 bundle')
        first = next(iter(self.bundle['loaded_starts'].values()))
        pose = first['state']['robot']['root_pose']
        embodiment.set_initial_pose(Pose(position_xyz=tuple(pose[:3]), rotation_xyzw=tuple(pose[3:])))
        self._support = embodiment
        return embodiment, teleop

    def build_scene(self, cfg):
        import isaaclab.sim as sim_utils
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        from dexbench_arena.assets import background_library  # noqa: F401
        from dexbench_arena.variations.db_h3_start_variation import H3StartVariation, H3StartVariationCfg

        starts = self.bundle['loaded_starts']
        first = next(iter(starts.values()))['state']
        assets = [self.asset_registry.get_asset_by_name('dexbench_hotel_fb_e4_260831')()]
        assets.append(Object(name='h3_light', object_type=ObjectType.BASE,
            spawner_cfg=sim_utils.DiskLightCfg(color=(1., 1., 1.), intensity=60000., radius=.3),
            initial_pose=Pose(position_xyz=(2., 1.5, 3.), rotation_xyzw=(0., 0., 0., 1.))))
        for item in self.bundle['assets']:
            pose = first['objects'][item['object_name']]['root_pose']
            asset = Object(name=item['object_name'], object_type=ObjectType.RIGID, usd_path=item['usd_path'],
                           initial_pose=Pose(position_xyz=tuple(pose[:3]), rotation_xyzw=tuple(pose[3:])))
            asset.disable_reset_pose()
            assets.append(asset)
        assets[-1].add_variation(H3StartVariation(starts, assets[-1].name,
            cfg=H3StartVariationCfg(head_joint_positions=dict(self.head_joint_positions))))
        return Scene(assets=assets)

    def _build_task(self, cfg):
        from dexbench_arena.tasks.extract_utensil import ExtractUtensilTask

        first = next(iter(self.bundle['loaded_starts'].values()))
        commands = dict(first['joint_targets'], **self.head_joint_positions)
        return ExtractUtensilTask(self.bundle['assets'], self.bundle['target'], commands, cfg.episode_length_s)

    available_tasks = {'extract_utensil': _build_task}

    def task_object_usd_path(self, task_object):
        """The utensil's pool package main USD (``packages`` in h3_objects.json), the file each runtime bundle
        binds by SHA-256. Read from the declaration, not the loaded bundle, so the manifest's asset-version
        contract (written without a build) and the episode's ``task_object`` record read the same sidecar."""
        return str(ASSETS_DIR / H3_PACKAGES[task_object.asset]['usd_path'])

    @classmethod
    def add_cli_args(cls, parser):
        super().add_cli_args(parser)
        parser.add_argument('--runtime_bundle', default='')
        parser.add_argument('--runtime_bundle_sha256', default='')
        parser.add_argument('--h3_object', choices=H3_OBJECT_AXIS.keys(), default='')
        parser.add_argument('--h3_inputs_dir', default=H3_INPUTS_DIR)
        parser.add_argument('--head_yaw_deg', type=float, default=5.)
        parser.add_argument('--head_pitch_deg', type=float, default=45.)
        parser.add_argument('--h3_drive_profile', choices=('collection_v1', 'main_default'), default='collection_v1')
        parser.set_defaults(**{name: getattr(H3EnvironmentCfg, name) for name in (
            'task', 'embodiment', 'anchor', 'rby1_control_hz', 'camera_scale', 'episode_length_s')})

    def apply_newton_sim_settings(self, env_cfg, cfg):
        super().apply_newton_sim_settings(env_cfg, cfg)
        physics = env_cfg.sim.physics
        physics.num_substeps = 8
        physics.collision_decimation = 1
        physics.solver_cfg.solver = 'newton'
        physics.solver_cfg.ls_iterations = 50
        physics.solver_cfg.nconmax = max(physics.solver_cfg.nconmax or 0, 2048)
        physics.solver_cfg.njmax = max(physics.solver_cfg.njmax or 0, 8192)
        physics.default_shape_cfg.ke, physics.default_shape_cfg.kd = solref_to_newton_ke_kd(.02, 2.)
        env_cfg.sim.gravity = tuple(self.bundle['gravity_m_s2'])
        ke, kd = solref_to_newton_ke_kd(.005, 1.)
        objects = '/(?:' + '|'.join(re.escape(a['object_name']) for a in self.bundle['assets']) + ')(?:/|$)'
        inertia_tokens = tuple('/' + a['object_name'] for a in self.bundle['assets'])
        install_isotropic_inertia(H3_UTENSIL_INERTIA_NAME, inertia_tokens, 'max')
        profile = self._support.action_config.joint_pos.grasp_profile
        requests = (
            ShapeContactRequest('h3_hands', self._support.hand_body_label_pattern, ke, kd, 1., match='body'),
            ShapeContactRequest('h3_utensils', objects, ke, kd, 1.),
            ShapeContactRequest('h3_left', self._support.support_grasp_body_pattern,
                                ke, kd, 1., match='body', friction=profile.friction),
        )
        for request in requests:
            install_shape_contact(request)
        install_rolling_friction('h3_utensils', objects, 2e-4)
        install_rolling_friction('h3_left', self._support.support_grasp_shape_pattern, profile.rolling_friction_m)
        setattr(env_cfg, NEWTON_SHAPE_OVERRIDES_ATTRIBUTE, {
            'shape_contact': {request.name: asdict(request) for request in requests},
            'rolling_friction': {
                'h3_utensils': dict(pattern=objects, mu_rolling=2e-4, condim=ROLLING_CONDIM),
                'h3_left': dict(pattern=self._support.support_grasp_shape_pattern,
                                mu_rolling=profile.rolling_friction_m, condim=ROLLING_CONDIM),
            },
            'isotropic_inertia': {H3_UTENSIL_INERTIA_NAME: {'tokens': list(inertia_tokens), 'mode': 'max'}},
            'inertia_policy': 'collection_isotropic_max',
            'robot_collision_filter': {
                'same_side_and_structural_pairs_excluded': env_cfg.scene.robot.spawn.bimanual_collision_filter,
                'left_thumb_opposition': False,
            },
        })
