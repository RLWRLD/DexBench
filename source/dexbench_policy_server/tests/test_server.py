"""The kit's HTTP contract, as the benchmark's client and the eval launcher's worker use it. Needs only the kit's own
dependencies (numpy, Pillow)."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import dexbench_policy_server
from dexbench_policy_server import ServedPolicy, make_server
from dexbench_policy_server.example import JOINT_NAMES

KIT = Path(__file__).resolve().parents[1]


def _request(port, path, token=None, payload=None):
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})}
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=headers,
                                     method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _png(height=180, width=320):
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_the_example_serves_the_contract_from_the_command_line(tmp_path):
    """What the worker and the client do: start it with a token file, wait for the flushed ready line, ask /info and
    /predict with the token; /health answers without one."""
    token = tmp_path / "token"
    token.write_text("secret\n")
    process = subprocess.Popen(
        [sys.executable, "-m", "dexbench_policy_server", "--policy", "dexbench_policy_server.example:HoldPose",
         "--checkpoint", str(tmp_path), "--port", "0", "--token-file", str(token),
         "--option", "cameras=camera_ego_left,camera_wrist_right", "--option", "frame_hw=180x320"],
        cwd=KIT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        line = process.stdout.readline()
        assert re.search("Server ready", line), line
        port = int(re.search(r"on 127\.0\.0\.1:(\d+)", line).group(1))
        assert _request(port, "/health") == (200, {"status": "ok"})
        assert _request(port, "/info")[0] == 401 and _request(port, "/info", "wrong")[0] == 401
        status, info = _request(port, "/info", "secret")
        assert status == 200 and info["joint_names"] == json.loads(JOINT_NAMES.read_text())
        assert info["camera_set"] == ["camera_ego_left", "camera_wrist_right"] and info["frame_hw"] == [180, 320]
        assert info["server"] == {"kit": "dexbench-policy-server", "version": dexbench_policy_server.__version__,
                                  "contract": dexbench_policy_server.CONTRACT}
        state = np.linspace(-1, 1, len(info["joint_names"])).tolist()
        request = {"images": {"camera_ego_left": _png(), "camera_wrist_right": _png()}, "state": state, "prompt": "p"}
        status, answer = _request(port, "/predict", "secret", request)
        assert status == 200 and np.allclose(answer["action"], [state] * info["action_horizon"])
        assert _request(port, "/predict", None, request)[0] == 401
        odd = urllib.request.Request(f"http://127.0.0.1:{port}/info", headers={"Authorization": "Bearer caf\u00e9"})
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(odd, timeout=30)
        assert refused.value.code == 401
        assert "error" in _request(port, "/predict", "secret", {**request, "state": state[:3]})[1]
    finally:
        process.terminate()
        process.wait(timeout=30)


class _Slow(ServedPolicy):
    """Counts the predicts running at once."""

    def __init__(self):
        self.running, self.most, self.lock = 0, 0, threading.Lock()

    def info(self):
        return {"joint_names": ["a", "b"], "action_horizon": 16, "frame_hw": [36, 64], "camera_set": ["camera_ego_left"]}

    def predict(self, images, state, prompt):
        with self.lock:
            self.running += 1
            self.most = max(self.most, self.running)
        time.sleep(0.2)
        with self.lock:
            self.running -= 1
        return np.zeros((16, 2))


def test_one_predict_at_a_time_however_many_simulators_share_the_server():
    """colocated+N: N simulators ask one server at once; the model never runs on two threads."""
    policy = _Slow()
    server = make_server(policy, "127.0.0.1", 0, None)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    request = {"images": {"camera_ego_left": _png(36, 64)}, "state": [0.0, 0.0], "prompt": "p"}
    answers = []
    threads = [threading.Thread(target=lambda: answers.append(_request(port, "/predict", None, request)))
               for _ in range(4)]
    for thread in threads:
        thread.start()
    time.sleep(0.05)
    assert _request(port, "/health") == (200, {"status": "ok"}), "health answers while a predict runs"
    for thread in threads:
        thread.join(timeout=30)
    server.shutdown()
    assert [status for status, _ in answers] == [200] * 4 and policy.most == 1


def test_a_policy_that_could_not_serve_is_refused_before_its_ready_line():
    class Short(_Slow):
        def info(self):
            return {**super().info(), "action_horizon": 8}

    with pytest.raises(ValueError, match="below 16"):
        make_server(Short(), "127.0.0.1", 0, None)

    class NumpyHorizon(_Slow):
        def info(self):
            return {**super().info(), "action_horizon": np.int64(16)}

    with pytest.raises(ValueError, match="not JSON"):
        make_server(NumpyHorizon(), "127.0.0.1", 0, None)


def test_the_version_is_the_packages():
    assert re.search(r'^version = "([^"]+)"', (KIT / "pyproject.toml").read_text(), re.M).group(1) == \
        dexbench_policy_server.__version__


def test_the_console_script_finds_a_policy_module_in_the_working_directory(tmp_path):
    """dexbench-policy-server --policy my_policy:MyPolicy, run beside my_policy.py: a console script puts its bin
    directory on the path, not the working directory (here a script in its own bin directory does the same, with the
    kit on PYTHONPATH)."""
    (tmp_path / "my_policy.py").write_text("from dexbench_policy_server.example import HoldPose as MyPolicy\n")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/dexbench-policy-server").write_text(
        "import sys\nfrom dexbench_policy_server.server import main\nsys.exit(main())\n")
    process = subprocess.Popen(
        [sys.executable, str(tmp_path / "bin/dexbench-policy-server"), "--policy", "my_policy:MyPolicy", "--port", "0"],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(KIT)}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)
    try:
        line = process.stdout.readline()
        assert "Server ready" in line, line + process.stdout.read() if process.poll() is not None else line
    finally:
        process.terminate()
        process.wait(timeout=30)


def test_predict_runs_on_the_thread_that_loaded_the_model(tmp_path):
    """Thread-local state set in load() (torch.set_grad_enabled(False), a CUDA stream) must hold in predict()."""
    (tmp_path / "local_policy.py").write_text(
        "import threading\nimport numpy as np\nfrom dexbench_policy_server import ServedPolicy\n"
        "LOCAL = threading.local()\n"
        "class Local(ServedPolicy):\n"
        "    def load(self, checkpoint, options):\n        LOCAL.loaded = True\n"
        "    def info(self):\n        return {'joint_names': ['a'], 'action_horizon': 16, 'frame_hw': [36, 64],"
        " 'camera_set': ['camera_ego_left']}\n"
        "    def predict(self, images, state, prompt):\n"
        "        assert getattr(LOCAL, 'loaded', False), 'predict ran on another thread than load'\n"
        "        return np.zeros((16, 1))\n")
    process = subprocess.Popen([sys.executable, "-m", "dexbench_policy_server", "--policy", "local_policy:Local",
                                "--port", "0"], cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(KIT)},
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        port = int(re.search(r"on 127\.0\.0\.1:(\d+)", process.stdout.readline()).group(1))
        request = {"images": {"camera_ego_left": _png(36, 64)}, "state": [0.0], "prompt": "p"}
        for _ in range(3):
            status, answer = _request(port, "/predict", None, request)
            assert status == 200 and "action" in answer, answer
    finally:
        process.terminate()
        process.wait(timeout=30)


def test_load_runs_on_the_main_thread_and_ctrl_c_stops_a_predict_at_once(tmp_path):
    """What needs the main thread works in load() (signal.signal, here), and SIGINT during a long predict stops the
    server at once rather than waiting the predict out."""
    (tmp_path / "main_policy.py").write_text(
        "import signal, time\nimport numpy as np\nfrom dexbench_policy_server import ServedPolicy\n"
        "class Main(ServedPolicy):\n"
        "    def load(self, checkpoint, options):\n        signal.signal(signal.SIGUSR1, signal.SIG_IGN)\n"
        "    def info(self):\n        return {'joint_names': ['a'], 'action_horizon': 16, 'frame_hw': [36, 64],"
        " 'camera_set': ['camera_ego_left']}\n"
        "    def predict(self, images, state, prompt):\n        time.sleep(30)\n        return np.zeros((16, 1))\n")
    process = subprocess.Popen([sys.executable, "-m", "dexbench_policy_server", "--policy", "main_policy:Main",
                                "--port", "0"], cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(KIT)},
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        line = process.stdout.readline()
        assert "Server ready" in line, line + (process.stdout.read() if process.poll() is not None else "")
        port = int(re.search(r"on 127\.0\.0\.1:(\d+)", line).group(1))
        request = {"images": {"camera_ego_left": _png(36, 64)}, "state": [0.0], "prompt": "p"}
        def ask():
            try:
                _request(port, "/predict", None, request)
            except OSError:
                pass

        threading.Thread(target=ask, daemon=True).start()
        time.sleep(1.0)
        process.send_signal(signal.SIGINT)
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)


def test_the_runner_stops_when_the_http_thread_has_ended():
    """A server whose HTTP thread died answers nothing; it must not stay up."""
    from dexbench_policy_server.server import MainThreadRunner

    with pytest.raises(RuntimeError, match="HTTP server's thread ended"):
        MainThreadRunner().run(serving=lambda: False)
