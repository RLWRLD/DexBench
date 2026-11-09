# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Keep fixed-root reset poses in sync with Newton's model and solver."""

from __future__ import annotations

from functools import wraps


def _runtime():
    from isaaclab_newton.physics import NewtonManager
    from newton import ModelFlags

    return NewtonManager, ModelFlags


def _copy_fixed_root_to_model(asset, model, selection, *, by_mask):
    """Copy selected cached link poses into the current model's real root view."""
    import warp as wp

    target = asset.root_view._get_attribute_array("joint_X_p", model, _slice=0)
    if not isinstance(target, wp.array) or target.ndim != 2 or target.shape[1] != 1:
        raise RuntimeError("Newton fixed-root compatibility requires one strided root per environment")
    source = asset.data.root_link_pose_w
    source = getattr(source, "warp", source)
    source = wp.to_torch(source)
    destination = wp.to_torch(target)[:, 0]
    if destination.shape != source.shape:
        raise RuntimeError("Newton fixed-root model and cached root shapes differ")

    if selection is None:
        destination.copy_(source)
        return
    selection = getattr(selection, "warp", selection)
    if isinstance(selection, wp.array):
        selection = wp.to_torch(selection)
    if by_mask:
        import torch

        torch.where(selection.bool()[:, None], source, destination, out=destination)
    else:
        destination[selection] = source[selection]


def _wrap_root_write(original, *, by_mask, from_link):
    if getattr(original, "_dexbench_fixed_root_reset", False):
        return original

    @wraps(original)
    def write(self, *args, **kwargs):
        if not self.root_view.is_fixed_base:
            return original(self, *args, **kwargs)

        skip_forward = kwargs.get("skip_forward", False)
        result = original(self, *args, **{**kwargs, "skip_forward": True})
        selection_name = "env_mask" if by_mask else "env_ids"
        selection = kwargs.get(selection_name)
        manager, flags = _runtime()
        _copy_fixed_root_to_model(self, manager.get_model(), selection, by_mask=by_mask)
        manager._solver.notify_model_changed(flags.JOINT_PROPERTIES)
        if not skip_forward:
            if by_mask:
                selection = self._ALL_ENV_MASK if selection is None else selection
            else:
                selection = self._resolve_env_ids(selection)
            self.data._reset_pose(from_link=from_link, **{selection_name: selection})
        return result

    write._dexbench_fixed_root_reset = True
    return write


def _install_on(asset_class):
    """Install once per class, preserving upstream metadata/signatures."""
    changed = False
    for frame in ("link", "com"):
        for selection in ("index", "mask"):
            name = f"write_root_{frame}_pose_to_sim_{selection}"
            original = getattr(asset_class, name)
            wrapped = _wrap_root_write(original, by_mask=selection == "mask", from_link=frame == "link")
            if wrapped is not original:
                setattr(asset_class, name, wrapped)
                changed = True
    return changed


def install_newton_fixed_root_reset_compat() -> bool:
    """Install the Newton-only root-write workaround after application startup."""
    from isaaclab_newton.assets.articulation.articulation import Articulation
    from isaaclab_newton.assets.rigid_object.rigid_object import RigidObject

    articulation_changed = _install_on(Articulation)
    rigid_changed = _install_on(RigidObject)
    return articulation_changed or rigid_changed
