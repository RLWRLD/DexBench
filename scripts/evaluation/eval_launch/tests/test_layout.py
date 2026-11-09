# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A layout is a placement shape: one-GPU bundles for one server and its simulators, every simulator on one GPU, and a
layout whose processes cannot fit in a GPU's memory refused before anything runs."""

from __future__ import annotations

import json

import pytest
import yaml

from eval_launch import launch
from eval_launch.layout import Slot, check_memory, parse_layout, stations


@pytest.mark.parametrize("value, gpus, server, sims", [
    ("colocated", 1, Slot(0, 0.5), [Slot(0, 0.5)]),
    ("server+1", 2, Slot(0, 1.0), [Slot(1, 1.0)]),
    ("server+2", 2, Slot(0, 1.0), [Slot(1, 0.5), Slot(1, 0.5)]),
    ("server+3", 2, Slot(0, 1.0), [Slot(1, 0.3333)] * 3),
    ({"sim_gpus": 2, "sims_per_gpu": 2}, 3, Slot(0, 1.0), [Slot(1, 0.5), Slot(1, 0.5), Slot(2, 0.5), Slot(2, 0.5)]),
])
def test_each_layout_is_one_gpu_bundles_with_the_server_first(value, gpus, server, sims):
    layout = parse_layout({"layout": value})
    assert layout.bundles() == [{"GPU": 1.0}] * gpus
    assert layout.server_slot() == server and layout.sim_slots() == sims
    for bundle in range(gpus):
        assert sum(s.gpu_share for s in [server, *sims] if s.bundle == bundle) <= 1.0


def test_no_layout_means_colocated_and_an_unknown_one_is_refused():
    assert parse_layout({}).name == "colocated" and parse_layout(None).sims == 1
    for value in ("server+0", "server+x", "two-gpu", {"sim_gpus": -1, "sims_per_gpu": 1}, {"sims_per_gpu": 2}):
        with pytest.raises(ValueError, match="servers.layout"):
            parse_layout({"layout": value})


def test_the_gpu_count_must_fill_whole_stations():
    assert stations(parse_layout({"layout": "server+2"}), 8) == 4
    with pytest.raises(ValueError, match="multiple of 2"):
        stations(parse_layout({"layout": "server+1"}), 3)


def test_a_cosmos_server_with_a_simulator_on_one_l40s_is_refused_and_on_its_own_gpu_is_not():
    """2026-10-23: the Cosmos3 server takes 33 of an L40S's 46 GB; an M1 simulator beside it ran out of memory."""
    servers = {"gpu_memory_gb": {"server": 33, "sim": 14}}
    with pytest.raises(ValueError, match="47 GB of a 46 GB GPU"):
        check_memory(parse_layout({"layout": "colocated"}), servers, 46)
    check_memory(parse_layout({"layout": "server+1"}), servers, 46)
    check_memory(parse_layout({"layout": "server+3"}), servers, 46)
    with pytest.raises(ValueError, match="56 GB"):
        check_memory(parse_layout({"layout": "server+4", "max_sims_per_gpu": 4}), servers, 46)


def test_several_simulators_on_a_gpu_need_their_memory_stated_and_colocated_does_not():
    check_memory(parse_layout({"layout": "colocated"}), {}, None)
    check_memory(parse_layout({"layout": "server+1"}), {}, None)
    with pytest.raises(ValueError, match="state servers.gpu_memory_gb.sim"):
        check_memory(parse_layout({"layout": "server+2"}), {}, 46)
    with pytest.raises(ValueError, match="server and sim"):
        check_memory(parse_layout({"layout": "server+1"}), {"gpu_memory_gb": {"kit": 10}}, 46)


def test_the_directory_backend_refuses_a_shared_gpu_layout_and_a_driver_launchs_output(tmp_path, capsys):
    """It runs one server and one simulator per GPU; and a launch keeps the backend it started with."""
    output = tmp_path / "launch"
    config = {"model": {"key": "gr00t", "checkpoint": "/a"}, "tasks": ["t"], "output": str(output),
              "servers": {"layout": "server+2", "recipe": {"command": "true"}}, "backend": {"kind": "k8s", "gpus": 2}}
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(SystemExit):
        launch.main([str(path), "--start"])
    assert "needs backend.kind: local or ray" in capsys.readouterr().err and not output.exists()
    (output / "state" / "drivers").mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps({"config": config, "units": []}))
    path.write_text(yaml.safe_dump({**config, "servers": {"recipe": {"command": "true"}}}))
    for mode in ("--local", "--resume", "--start"):
        with pytest.raises(SystemExit):
            launch.main([str(path), mode])
        assert "is a driver's launch" in capsys.readouterr().err
    assert not (output / "queue").exists()


def test_a_layout_is_refused_past_the_simulators_per_gpu_cap():
    """Three simulators on a GPU pass the default cap; four are refused, naming the cap and the count, unless the
    config raises the cap."""
    assert parse_layout({"layout": "server+3"}).sims_per_gpu == 3
    with pytest.raises(ValueError, match=r"puts 4 simulators on one GPU, more than servers.max_sims_per_gpu \(3\)"):
        parse_layout({"layout": {"sim_gpus": 2, "sims_per_gpu": 4}})
    assert parse_layout({"layout": "server+4", "max_sims_per_gpu": 4}).sims_per_gpu == 4


def test_the_server_and_several_simulators_share_one_gpu_when_sim_gpus_is_zero():
    """One server and three simulators on one GPU, a quarter each; the memory check counts the server with them, so it
    must be stated."""
    layout = parse_layout({"layout": {"sim_gpus": 0, "sims_per_gpu": 3}})
    assert layout.name == "colocated+3" and layout.gpus == 1 and layout.sims == 3
    assert layout.server_slot() == Slot(0, 0.25) and layout.sim_slots() == [Slot(0, 0.25)] * 3
    assert stations(layout, 4) == 4
    with pytest.raises(ValueError, match=r"and .server, when they share its GPU"):
        check_memory(layout, {"gpu_memory_gb": {"sim": 10}}, 46)
    with pytest.raises(ValueError, match="1 server.s. and 3 simulator.s. on GPU 0 of each station: 47 GB of a 46 GB"):
        check_memory(layout, {"gpu_memory_gb": {"server": 17, "sim": 10}}, 46)
    check_memory(parse_layout({"layout": {"sim_gpus": 0, "sims_per_gpu": 2}}),
                 {"gpu_memory_gb": {"server": 17, "sim": 10}}, 46)
