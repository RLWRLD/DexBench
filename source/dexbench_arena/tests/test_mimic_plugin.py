# SPDX-License-Identifier: Apache-2.0
"""The Mimic plugin (``dexbench_arena.mimic_plugin``): a tree without the ``datagen`` package builds every task's base
class and registers no data-generation embodiment, and a ``datagen`` package that fails on an import of its own is
reported, not taken for an absent one. Kit-free: the packages here are stand-ins written to a temporary directory."""

from __future__ import annotations

import sys
import textwrap

import pytest

from dexbench_arena import mimic_plugin


class BaseTask:
    pass


class OtherTask:
    pass


def _package(tmp_path, monkeypatch, init: str, **modules: str) -> None:
    """A stand-in datagen package ``fake_datagen`` (with ``modules`` beside it) that the plugin imports by name."""
    (tmp_path / "fake_datagen").mkdir()
    (tmp_path / "fake_datagen" / "__init__.py").write_text(textwrap.dedent(init))
    for name, text in modules.items():
        (tmp_path / f"{name}.py").write_text(textwrap.dedent(text))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(mimic_plugin, "DATAGEN", "fake_datagen")
    for name in ("fake_datagen", *modules):
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_without_datagen_every_task_is_its_base_and_no_embodiment_registers(monkeypatch):
    monkeypatch.setitem(sys.modules, mimic_plugin.DATAGEN, None)
    assert mimic_plugin.datagen() is None
    assert mimic_plugin.mimic_task_class(BaseTask) is BaseTask
    mimic_plugin.register_datagen_embodiments()
    with pytest.raises(ValueError, match="--mimic_segments needs the Mimic code"):
        mimic_plugin.require_datagen("--mimic_segments")


def test_a_datagen_package_that_fails_on_another_import_still_raises(tmp_path, monkeypatch):
    _package(tmp_path, monkeypatch, "import fake_datagen_missing_dependency\n")
    with pytest.raises(ModuleNotFoundError) as error:
        mimic_plugin.datagen()
    assert error.value.name == "fake_datagen_missing_dependency"
    with pytest.raises(ModuleNotFoundError):
        mimic_plugin.mimic_task_class(BaseTask)


def test_with_datagen_a_task_is_its_mimic_subclass_and_the_embodiments_register(tmp_path, monkeypatch):
    base = f"{__name__}.BaseTask"
    _package(tmp_path, monkeypatch, f"""
        MIMIC_TASKS = {{{base!r}: "fake_mimic_tasks:MimicTask", {f"{__name__}.OtherTask"!r}: "fake_mimic_tasks:NotOne"}}
        EMBODIMENT_MODULES = ("fake_embodiment",)
        """,
        fake_mimic_tasks=f"""
        import {__name__} as tests
        class MimicTask(tests.BaseTask):
            pass
        class NotOne:
            pass
        """,
        fake_embodiment="REGISTERED = True\n")
    mimic = mimic_plugin.mimic_task_class(BaseTask)
    assert issubclass(mimic, BaseTask) and mimic.__name__ == "MimicTask"
    with pytest.raises(TypeError, match="not a subclass"):
        mimic_plugin.mimic_task_class(OtherTask)

    class Unlisted:
        pass

    with pytest.raises(LookupError, match="names no Mimic subclass"):
        mimic_plugin.mimic_task_class(Unlisted)
    assert "fake_embodiment" not in sys.modules
    mimic_plugin.register_datagen_embodiments()
    assert sys.modules["fake_embodiment"].REGISTERED
    mimic_plugin.require_datagen("--mimic_segments")
