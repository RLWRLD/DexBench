# SPDX-License-Identifier: Apache-2.0
"""Optional Isaac Lab Mimic support: the ``datagen`` package, found by its name; a tree without it builds every task
without Mimic."""

from __future__ import annotations

import importlib
from types import ModuleType

DATAGEN = "dexbench_arena.datagen"
"""The package's import name."""


def datagen() -> ModuleType | None:
    """The ``datagen`` package; None when this tree does not have it."""
    try:
        return importlib.import_module(DATAGEN)
    except ModuleNotFoundError as error:
        if error.name != DATAGEN:
            raise
        return None


def mimic_task_class(base: type) -> type:
    """The class a factory builds for task class ``base``: its Mimic subclass, which ``datagen.MIMIC_TASKS`` names
    under ``base``'s ``"module.QualifiedName"`` as ``"module:Name"``, or ``base`` itself without the package."""
    package = datagen()
    if package is None:
        return base
    key = f"{base.__module__}.{base.__qualname__}"
    entry = package.MIMIC_TASKS.get(key)
    if entry is None:
        raise LookupError(f"{DATAGEN}.MIMIC_TASKS names no Mimic subclass for {key}")
    module, _, name = entry.partition(":")
    cls = getattr(importlib.import_module(module), name)
    if not (isinstance(cls, type) and issubclass(cls, base)):
        raise TypeError(f"{DATAGEN}.MIMIC_TASKS[{key!r}] is {entry}, which is not a subclass of {key}")
    return cls


def require_datagen(what: str) -> None:
    """Refuse ``what``, an option that only the Mimic code reads, in a tree without the ``datagen`` package."""
    if datagen() is None:
        raise ValueError(f"{what} needs the Mimic code ({DATAGEN}), which this tree does not have")


def register_datagen_embodiments() -> None:
    """Import the modules ``datagen.EMBODIMENT_MODULES`` names, so their ``@register_asset`` decorators run; nothing
    without the package."""
    package = datagen()
    for module in () if package is None else package.EMBODIMENT_MODULES:
        importlib.import_module(module)
