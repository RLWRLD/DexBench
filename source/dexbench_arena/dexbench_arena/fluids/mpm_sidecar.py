"""DB-H2's water as a standalone Newton implicit MPM model beside the robot's physics."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass

import newton

import numpy as np
import warp as wp

logger = logging.getLogger(__name__)


@dataclass
class ImplicitMPMSidecarCfg:
    """Configuration of the implicit MPM water (the patch's ``ImplicitMPMSidecarCfg``, the fields this model reads)."""

    voxel_size: float = 0.005
    """MPM grid voxel size in meters."""

    grid_type: str = "fixed"
    """MPM grid type. Use ``fixed`` for deterministic allocation and graph-capture friendliness."""

    grid_padding: int = 8
    """Padding cells around the active particle domain for fixed grids."""

    max_active_cell_count: int = 1 << 18
    """Maximum active grid cells allocated by the MPM solver."""

    max_iterations: int = 80
    """Maximum nonlinear/linear iterations used by the implicit MPM solve."""

    tolerance: float = 1.0e-4
    """Solver convergence tolerance."""

    solver: str = "gauss-seidel"
    """Linear solver used by SolverImplicitMPM."""

    transfer_scheme: str = "apic"
    """Particle-grid transfer scheme."""

    integration_scheme: str = "pic"
    """MPM integration scheme."""

    strain_basis: str = "P0"
    """Basis used for strain/stress fields."""

    collider_basis: str = "Q1"
    """Basis used for collider fields."""

    velocity_basis: str = "Q1"
    """Basis used for velocity fields."""

    density: float = 1000.0
    """Particle material density in kg/m^3 for the built-in box emitter."""

    viscosity: float = 0.5
    """Per-particle viscosity assigned after model finalization."""

    tensile_yield_ratio: float = 1.0
    """Per-particle tensile yield ratio assigned after model finalization."""

    friction: float = 0.0
    """Per-particle friction assigned after model finalization."""

    particles_per_cell: float = 2.0
    """Particle sampling density per MPM grid cell for the built-in box emitter."""

    particle_jitter_scale: float = 1.0
    """Jitter multiplier applied to the generated particle radius."""

    particle_box_min: tuple[float, float, float] = (-0.05, -0.05, 0.5)
    """Minimum XYZ corner for the built-in particle box emitter."""

    particle_box_max: tuple[float, float, float] = (0.05, 0.05, 0.6)
    """Maximum XYZ corner for the built-in particle box emitter."""


SETUP_COLLIDER_ARGS = ("collider_meshes", "collider_body_ids", "collider_margins", "collider_friction")
"""What ``setup_mpm_colliders`` hands ``SolverImplicitMPM.setup_collider``, by the pinned newton's (1.8) names. A
newton that does not take one of them is refused rather than given fewer: a dropped margin or friction would change
the water's contact unseen."""


def emit_particle_box(mpm_builder, mpm_cfg) -> None:
    """Emit a box of MPM particles from config."""
    voxel_size = float(mpm_cfg.voxel_size)
    particles_per_cell = float(mpm_cfg.particles_per_cell)
    density = float(mpm_cfg.density)
    box_min = np.array(mpm_cfg.particle_box_min, dtype=float)
    box_max = np.array(mpm_cfg.particle_box_max, dtype=float)

    if voxel_size <= 0.0:
        raise ValueError("MPM sidecar: voxel_size must be positive.")
    if particles_per_cell <= 0.0:
        raise ValueError("MPM sidecar: particles_per_cell must be positive.")
    if np.any(box_max <= box_min):
        raise ValueError("MPM sidecar: particle_box_max must be greater than particle_box_min.")

    box_size = box_max - box_min
    resolution = np.maximum(np.ceil(particles_per_cell * box_size / voxel_size).astype(int), 1)
    cell_size = box_size / resolution
    cell_volume = float(np.prod(cell_size))
    radius = float(np.max(cell_size) * 0.5)
    mass = float(cell_volume * density)
    jitter = float(mpm_cfg.particle_jitter_scale) * radius

    mpm_builder.add_particle_grid(
        pos=wp.vec3(float(box_min[0]), float(box_min[1]), float(box_min[2])),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=int(resolution[0]) + 1,
        dim_y=int(resolution[1]) + 1,
        dim_z=int(resolution[2]) + 1,
        cell_x=float(cell_size[0]),
        cell_y=float(cell_size[1]),
        cell_z=float(cell_size[2]),
        mass=mass,
        jitter=jitter,
        radius_mean=radius,
    )


def set_material_parameters(model, mpm_cfg) -> None:
    """Set per-particle MPM material parameters from config (the attributes ``SolverImplicitMPM``'s
    ``register_custom_attributes`` adds; a model without them fails here rather than keeping newton's defaults)."""
    for name in ("viscosity", "tensile_yield_ratio", "friction"):
        getattr(model.mpm, name).fill_(float(getattr(mpm_cfg, name)))


class StandaloneMPMHost:
    """One standalone fluid's model, solver and state, and its collider bookkeeping."""

    _mpm_model = None
    _mpm_solver = None
    _mpm_state_0 = None
    _mpm_enabled = False

    _emit_mpm_particle_box = staticmethod(emit_particle_box)
    _set_mpm_material_parameters = staticmethod(set_material_parameters)

    @classmethod
    def get_mpm_state_0(cls):
        """The fluid's state, or None before it is built."""
        return cls._mpm_state_0

    @classmethod
    def setup_mpm_colliders(
        cls,
        collider_meshes: list,
        collider_body_ids: list | None = None,
        collider_margins: list | None = None,
        collider_friction: list | None = None,
    ) -> None:
        """Register explicit collider meshes (``wp.Mesh`` in world space) with the fluid's solver."""
        if not cls._mpm_enabled or cls._mpm_solver is None:
            raise RuntimeError("MPM sidecar: the fluid must be built before setup_mpm_colliders().")
        missing = sorted(set(SETUP_COLLIDER_ARGS) - set(inspect.signature(cls._mpm_solver.setup_collider).parameters))
        if missing:
            raise RuntimeError(f"MPM sidecar: newton {newton.__version__}'s SolverImplicitMPM.setup_collider takes no "
                               f"{', '.join(missing)}; the water is written against newton 1.8's (SETUP_COLLIDER_ARGS)")
        cls._mpm_solver.setup_collider(collider_meshes=collider_meshes, collider_body_ids=collider_body_ids,
                                       collider_margins=collider_margins, collider_friction=collider_friction)
        logger.debug("MPM sidecar: registered %d explicit collider meshes.", len(collider_meshes))

    @classmethod
    def _step_mpm_sidecar(cls, rigid_state, dt: float) -> None:  # noqa: ARG003 (rigid_state: the patch's signature)
        """Advance the fluid by ``dt``: the implicit solve, then the per-particle projection out of the colliders."""
        if not cls._mpm_enabled or cls._mpm_solver is None or cls._mpm_state_0 is None:
            return
        cls._mpm_solver.step(cls._mpm_state_0, cls._mpm_state_0, control=None, contacts=None, dt=dt)
        cls._mpm_solver.project_outside(cls._mpm_state_0, cls._mpm_state_0, dt)
