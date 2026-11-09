"""Known package versions never bypass comparison merely because no physical epoch opted in."""
import pytest

from dexbench_arena.tasks.object_axis import TaskObject, asset_major_problem


@pytest.mark.parametrize("versions,renames,refused", [
    ({"source": "1.2.0", "current": "2.0.0"}, (), True),
    ({"source": "1.2.0", "current": "2.0.0"}, ((1, 2, "CHANGELOG.md"),), False),
    ({"source": "1.2.0", "current": "3.0.0"}, ((1, 2, "CHANGELOG.md"), (2, 3, "CHANGELOG.md")), True),
    ({"source": "2.0.0", "current": "1.2.0"}, ((1, 2, "CHANGELOG.md"),), True),
    ({"source": "2.0.0", "episode": None, "current": "2.0.0"}, (), True),
    ({"source": "2.0.0", "current": None}, (), True),
    ({"source": None, "episode": None, "current": "2.0.0"}, (), False),
    ({"source": None, "episode": "2.0.0", "current": "3.0.0"}, (), True),
    ({"source": None, "current": None}, (), False),
    ({"source": "invalid", "current": "2.0.0"}, (), True),
])
def test_known_versions_and_exact_directional_rename_exceptions(versions, renames, refused):
    assert (asset_major_problem(None, versions, rename_only=renames) is not None) is refused


@pytest.mark.parametrize("renames", [((1, 2, ""),), ((2, 1, "evidence"),),
                                    ((1, 2, "evidence"), (1, 2, "duplicate"))])
def test_rename_exceptions_need_exact_pairs_and_evidence(renames):
    with pytest.raises(ValueError, match="rename-only"):
        TaskObject("sample", "sample", "train", "sample", (1.0,), benchmark_asset_major_renames=renames)
