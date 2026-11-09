# SPDX-License-Identifier: Apache-2.0
"""Exercise the external runtime on CUDA without downloading robot or scene assets."""
from __future__ import annotations

import json
import sys

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from dexbench_arena.app_launch import launch_isaac_app, run_then_close_app

parser = get_isaaclab_arena_cli_parser()
parser.add_argument('--steps', type=int, default=120)
parser.set_defaults(presets='physx')
args = parser.parse_args()
if args.steps < 120:
    parser.error('--steps must be at least 120 to check settled contact')
if not str(args.device).startswith('cuda'):
    parser.error('--device must select CUDA')
app = launch_isaac_app(args).app


def main() -> None:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    import dexbench_arena

    assert torch.cuda.is_available(), 'CUDA unavailable'
    matrix = torch.ones((256, 256), device=args.device)
    assert torch.all(matrix @ matrix == 256).item(), 'CUDA matrix multiplication failed'
    torch.cuda.synchronize()
    dexbench_arena.register_components()
    registered = sorted(k for k in EnvironmentRegistry().get_all_keys() if k.startswith('dexbench_'))
    assert registered, 'DexBench environment registration failed'
    assert ArenaEnvBuilder is not None

    physics = {'physx': PhysxCfg, 'newton': NewtonCfg}[args.presets]()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0/60.0, device=args.device, physics=physics))
    floor = sim_utils.CuboidCfg(size=(4.0, 4.0, 0.1), collision_props=sim_utils.CollisionPropertiesCfg())
    floor.func('/World/Floor', floor, translation=(0.0, 0.0, -0.05))
    box = RigidObject(RigidObjectCfg(
        prim_path='/World/Box',
        spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    ))
    sim.reset()
    start_z = float(box.data.root_pos_w.torch[0, 2])
    for _ in range(args.steps):
        sim.step(render=False)
        box.update(sim.get_physics_dt())
    positions = box.data.root_pos_w.torch
    assert positions.is_cuda, f'Physics state is not on CUDA: {positions.device}'
    assert torch.isfinite(positions).all().item(), 'Non-finite physics state'
    final_z = float(positions[0, 2])
    assert start_z > 0.9 and 0.07 < final_z < 0.14, (start_z, final_z)
    print('SIMULATION_GPU_SMOKE=' + json.dumps({
        'gpu': torch.cuda.get_device_name(), 'backend': str(args.presets),
        'steps': args.steps, 'start_z': start_z, 'final_z': final_z,
        'state_device': str(positions.device), 'registered_environments': len(registered),
    }), flush=True)


if __name__ == '__main__':
    sys.exit(run_then_close_app(app, main))
