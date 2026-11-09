# SPDX-License-Identifier: Apache-2.0
"""Configuration of the VBD polybag asset (film + cotton) for Newton scenes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets.deformable_object.deformable_object_cfg import DeformableObjectCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .polybag_object import PolybagObject


@configclass
class PolybagVisualCfg(SpawnerCfg):
    """The prim a polybag is drawn with: an Xform and, under it, the film's simulation mesh (film and cotton
    bag) or the asset's textured mesh and its material (tet bag, ``tet_template.py``)."""

    func: Callable | str = "{DIR}.polybag_object:spawn_polybag_visual"

    usd_path: str = MISSING
    """The polybag package's USD (``object_library/polybag_bubble_cotton_deformable`` or
    ``object_library/polybag_cotton_tet_deformable``)."""

    color: Sequence[float] = (0.35, 0.65, 0.40)
    """RGB display color of the film (a tet bag is drawn with its own material)."""

    diffuse_texture: str | None = None
    """A tet bag's albedo texture in place of the one its material reads (an image file); None keeps the asset's.
    ``diffuse_tint_rgb`` recolours this one."""

    diffuse_tint_rgb: tuple[int, int, int] | None = None
    """A tet bag's colour (0-255 sRGB): its albedo texture recoloured to it, the label and barcode left as they
    are (``tint_albedo``). None draws the asset's own texture. The film and cotton bag takes ``color``."""


@configclass
class PolybagObjectCfg(DeformableObjectCfg):
    """A polybag simulated with Newton's VBD solver: film and cotton built by the canonical loader
    (``canonical/``), or one tet soft body (``tet_template.py``), as the asset is."""

    class_type: type[PolybagObject] | str = "{DIR}.polybag_object:PolybagObject"

    cloning_contexts: tuple[str | type, ...] | None = ("isaaclab_newton.cloner:NewtonReplicateContext",)
    """Newton replication: the bag is added to every world's builder by the asset's builder hook."""

    spawn: PolybagVisualCfg = MISSING

    cotton_youngs_modulus: float | None = 10_000.0
    """Young's modulus of the cotton filling [Pa]; None keeps the asset's 2 kPa."""

    cotton_poisson_ratio: float | None = 0.45
    """Poisson ratio of the cotton filling; None keeps the asset's 0.1."""

    film_bend_scale: float | None = None
    """Factor on the film's bending stiffness and damping; None keeps the asset's (edgeKe 1.07)."""

    film_rest_bend_angles: str | None = "restShape"
    """The film's bending rest angles: "restShape" keeps the folds of the delivered mesh (what the asset was
    tuned with, on Newton 1.5), "flat" bends every edge toward flat (what Newton 1.6+ does to a film that
    authors no ``physics:restBendAnglesDefault``, as this asset), None leaves it to the Newton in use."""


    tet_youngs_modulus: float | None = None
    """Young's modulus of the tet bag [Pa]; None keeps the asset's 15 kPa."""

    tet_poisson_ratio: float | None = None
    """Poisson ratio of the tet bag, in [0, 0.5); None keeps the asset's 0.15."""

    tet_k_damp: float | None = None
    """Damping of the tet bag's tets; None keeps the asset's 30."""

    tet_contact_ke: float | None = None
    """The bag's contact stiffness [N/m] (``Model.soft_contact_ke``: its self-contact, and its half of a contact
    with a collider); None keeps Newton's 1e3."""

    tet_contact_kd: float | None = None
    """The bag's contact damping (``Model.soft_contact_kd``); None keeps Newton's 10."""

    tet_contact_mu: float | None = None
    """The bag's friction (``Model.soft_contact_mu``); None keeps Newton's 0.5."""

    tet_collider_ke: float | None = None
    """Every collider's contact stiffness against the bag [N/m], the robot's hands included; None keeps the
    delivery's 1e4. A body-particle contact's stiffness and damping are the mean of the two sides'."""

    tet_collider_kd: float | None = None
    """Every collider's contact damping against the bag; None keeps the delivery's 10."""

    tet_collider_mu: float | None = None
    """Every collider's friction against the bag; None keeps the delivery's 0.8. A body-particle contact's
    friction is the root of the product of the two sides'."""

    tet_collider_mu_by_path: tuple[tuple[str, float], ...] | None = None
    """(pattern, friction) pairs: a collider whose prim path matches a pattern (``re.search``, the first that
    matches) takes that friction against the bag in place of ``tet_collider_mu``. A pattern that matches no
    collider is refused when the solver is built. None: every collider takes ``tet_collider_mu``."""

    tet_size_scale: tuple[float, float, float] | None = None
    """The tet bag's (x, y, z) size over the asset's, z its thickness, its points scaled before the import
    (``tet_template.size_values``); None keeps the asset's."""

    mass_kg: float | None = None
    """The bag's mass [kg]: every particle's mass scaled by one factor so they sum to it
    (``add_polybag_entry_to_builder``); None keeps the asset's. Copies of a template share it only through
    their own cfg, so bags of one template can weigh differently."""


    vbd_substeps: int = 16
    """Physics substeps per step, both solvers. The asset was verified at 128 x 80; the cotton bag
    bursts below 16 substeps."""

    mjwarp_substeps: int = 8
    """MJWarp substeps per physics step (the robot); divides ``vbd_substeps``. 8 at 60 Hz is L2's 1/480 s
    MJWarp step: MJWarp runs in every other of the bag's substeps, and the bag reads the robot's bodies
    interpolated in between. ``vbd_substeps``: every substep."""

    vbd_iterations: int = 6
    """VBD iterations per substep."""

    frame_truncation: str | None = "init"
    """Self-contact truncation in the bag's own frame instead of SolverVBD's isotropic DAT clamp, which caps
    the bag's travel at ``margin x 0.85 x 0.5 x 60 x substeps`` (a free fall at 0.2 m/s at 8 substeps).
    "init": on each substep's initial guess; "all": on every truncation call; None: SolverVBD's own."""

    coupling_mode: str = "one_way"
    """How the bag and MJWarp's bodies (the robot, rigid parcels) push on each other: "one_way", the bodies
    move the bag and feel nothing back; "two_way", the bag's contact reactions go into the bodies' forces
    before MJWarp steps (``CoupledMJWarpVBDSolverCfg.coupling_mode``). Two-way needs MJWarp in every
    substep (``mjwarp_substeps == vbd_substeps``)."""

    fall_height_m: float = 0.8
    """The farthest the bag can fall in the scene [m]; with the stepping it sets the soft-contact band, so
    that no particle crosses a collider between two substeps."""

    soft_contacts_per_particle: int = 8
    """Capacity of the scene's soft-contact buffer per bag particle. Newton sizes it by default to every
    particle x every shape (18 million in the E2 cell) and SolverVBD launches its body-particle contact
    kernels over that capacity, not over the contacts found, in every iteration and colour sweep. A
    particle touches a few colliders at once (the belt, a fingertip or two)."""

    full_surface_contact: bool = False
    """Newton's full-surface rigid-soft contact: the bag's surface edges and faces also touch the colliders,
    sampled from each collider's signed-distance field, not only its particles. A tet bag's surface particles
    are ~2.5 cm apart, wider than a fingertip, which would pass between them; its delivery was validated with
    it on. Every mesh collider within ``collider_reach_m`` and every robot mesh gets an SDF (``sdf_voxel_m``)."""

    contact_log_barrier: bool = False
    """SolverVBD's ``rigid_soft_contact_use_log_barrier``: the body-particle normal contact as the particles' own C2
    log barrier, which stiffens without bound as a collider closes in, instead of the linear penalty (False, Newton
    1.6.1's)."""

    sdf_voxel_m: float = 0.002
    """Voxel size of the colliders' SDFs that full-surface contact samples [m]."""

    box_colliders: tuple[str, ...] = ()
    """Box-shaped mesh colliders (a label regex each, e.g. the conveyor belt) whose contact with the bag goes
    through an analytic box of the same size instead (``particle_boxes.add_particle_boxes``; the robot still sees the
    mesh). A mesh contact finds a particle only within the soft-contact band of its surface: one pushed deeper
    (a finger pressing the bag onto the belt) is never pushed back out and stays caught inside. A box's signed
    distance holds everywhere in it, as the delivery's validation tables and floors were."""

    collider_reach_m: float | None = 1.5
    """Static colliders whose bounds stay farther than this from the bag's initial bounds [m] do not collide
    with its particles (every body's colliders always do). Newton checks every particle against every
    particle-colliding shape each substep, mesh shapes with a BVH query: in the E2 cell, 18704 particles x
    689 shapes. None: every static collider."""
