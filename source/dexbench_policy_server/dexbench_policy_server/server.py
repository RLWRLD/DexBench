"""The HTTP server and its command line: the benchmark's contract around one :class:`ServedPolicy`."""

from __future__ import annotations

import argparse
import base64
import hmac
import importlib
import io
import json
import os
import queue
import sys
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dexbench_policy_server.policy import ServedPolicy

VERSION = "0.1.0"
"""This kit's version (``pyproject.toml`` states the same); ``/info`` reports it."""
CONTRACT = 1
"""The HTTP contract's version, reported at ``/info``: a change a client must know about raises it."""
READY_LINE = "Server ready"
"""What the ready line starts with: the eval launcher's ``servers.recipe.ready`` default."""
INFO_KEYS = ("joint_names", "action_horizon", "frame_hw", "camera_set")
"""What a policy's ``info()`` must state; ``action_representation`` is optional (absolute)."""
MIN_ACTION_HORIZON = 16
"""The benchmark executes the first 16 actions of each answer."""


def check_info(info: dict[str, Any]) -> dict[str, Any]:
    """``info`` as the client will read it, refused at start when it could not serve: a missing key, a horizon below
    :data:`MIN_ACTION_HORIZON`, a frame that is not ``[height, width]``, or no camera."""
    missing = [key for key in INFO_KEYS if key not in info]
    if missing:
        raise ValueError(f"the policy's info() lacks {missing}; it states {list(INFO_KEYS)}")
    if int(info["action_horizon"]) < MIN_ACTION_HORIZON:
        raise ValueError(f"action_horizon {info['action_horizon']} is below {MIN_ACTION_HORIZON}: the benchmark executes "
                         f"the first {MIN_ACTION_HORIZON} actions of each answer")
    if len(info["frame_hw"]) != 2:
        raise ValueError(f"frame_hw is [height, width], not {info['frame_hw']!r}")
    if not info["camera_set"] or not info["joint_names"]:
        raise ValueError("camera_set and joint_names name at least one camera and one joint")
    try:
        info = json.loads(json.dumps(info))
    except (TypeError, ValueError) as error:
        raise ValueError(f"the policy's info() is not JSON (use plain int, float, str and lists): {error}") from None
    return {**info, "server": {"kit": "dexbench-policy-server", "version": VERSION, "contract": CONTRACT}}


def decode_png(data: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))


class MainThreadRunner:
    """Runs the calls submitted to it on the thread that calls :meth:`run` (the main thread, in :func:`serve`), one at
    a time, in order: :func:`make_server`'s ``runner``."""

    def __init__(self) -> None:
        self.calls: queue.Queue = queue.Queue()

    def submit(self, function, *args) -> Future:
        future: Future = Future()
        self.calls.put((function, args, future))
        return future

    def run(self, serving: Callable[[], bool] | None = None) -> None:
        """Run submitted calls until the process is stopped (a KeyboardInterrupt raised in a call stops it too), or
        until ``serving()`` says the HTTP server's thread has ended, which raises: a server that answers nothing must
        not stay up."""
        while True:
            try:
                function, args, future = self.calls.get(timeout=0.5)
            except queue.Empty:
                if serving is not None and not serving():
                    raise RuntimeError("the HTTP server's thread ended; the server stops") from None
                continue
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(function(*args))
            except Exception as error:
                future.set_exception(error)
            except BaseException:
                future.set_exception(RuntimeError("the server is stopping"))
                raise


def make_server(policy: ServedPolicy, host: str, port: int, token: str | None,
                runner: MainThreadRunner | ThreadPoolExecutor | None = None) -> ThreadingHTTPServer:
    """The HTTP server for a loaded ``policy`` on ``host:port`` (not yet serving): ``serve_forever`` it. Every predict
    is handed to ``runner``, one at a time: :func:`serve` passes a :class:`MainThreadRunner`, whose calls run on the
    thread that ran ``load``; left out, a one-thread executor of its own. Its ``policy_info`` is the checked info
    ``/info`` answers."""
    info = check_info(policy.info())
    shape = (int(info["action_horizon"]), len(info["joint_names"]))
    runner = runner or ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy")

    class Handler(BaseHTTPRequestHandler):
        def _answer(self, status: int, body: dict) -> None:
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _authorized(self) -> bool:
            given = self.headers.get("Authorization", "").encode("latin-1", "replace")
            if token is None or hmac.compare_digest(given, f"Bearer {token}".encode()):
                return True
            self._answer(401, {"error": "missing or wrong token"})
            return False

        def do_GET(self):
            if self.path == "/health":
                self._answer(200, {"status": "ok"})
            elif not self._authorized():
                return
            elif self.path == "/info":
                self._answer(200, info)
            else:
                self._answer(404, {"error": f"no {self.path}"})

        def do_POST(self):
            if not self._authorized():
                return
            if self.path != "/predict":
                self._answer(404, {"error": f"no {self.path}"})
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                images = {key: decode_png(value) for key, value in request["images"].items()}
                state = np.asarray(request["state"], dtype=np.float64)
                answer = runner.submit(policy.predict, images, state, request["prompt"]).result()
                actions = np.asarray(answer, dtype=np.float64)
                if actions.shape != shape or not np.isfinite(actions).all():
                    raise ValueError(f"the policy answered {actions.shape}, expected {shape} finite values")
                self._answer(200, {"action": actions.tolist()})
            except Exception as error:
                self._answer(200, {"error": repr(error)})

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.policy_info = info
    return server


def load_policy(spec: str) -> ServedPolicy:
    """The policy class ``module:Class`` names, made with no arguments. The module is looked for in the working
    directory first, as ``python -m dexbench_policy_server`` does: the ``dexbench-policy-server`` console script puts
    only its own bin directory on the path, so a ``my_policy.py`` beside the command would not be found."""
    module_name, separator, class_name = spec.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError(f"--policy is module:Class, not {spec!r}")
    if not getattr(sys.flags, "safe_path", False) and "" not in sys.path and os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    cls = getattr(importlib.import_module(module_name), class_name)
    if not (isinstance(cls, type) and issubclass(cls, ServedPolicy)):
        raise TypeError(f"{spec} is not a ServedPolicy subclass")
    return cls()


def serve(policy: ServedPolicy, checkpoint: str | None, host: str, port: int, token: str | None,
          options: dict[str, str] | None = None) -> None:
    """Load ``policy`` on this (the main) thread, print the ready line (flushed) and answer requests until the process
    is stopped: the HTTP server on a thread of its own, the model's calls back here (:class:`MainThreadRunner`)."""
    policy.load(checkpoint, dict(options or {}))
    runner = MainThreadRunner()
    server = make_server(policy, host, port, token, runner)
    info = server.policy_info
    print(f"{READY_LINE} on {host}:{server.server_address[1]} ({len(info['joint_names'])} joints, horizon "
          f"{info['action_horizon']}, cameras {list(info['camera_set'])}, token {'required' if token else 'not required'}, "
          f"dexbench-policy-server {VERSION})", flush=True)
    http = threading.Thread(target=server.serve_forever, name="http", daemon=True)
    http.start()
    try:
        runner.run(serving=http.is_alive)
    finally:
        server.shutdown()
        server.server_close()


def _option(text: str) -> tuple[str, str]:
    key, separator, value = text.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError(f"--option is key=value, not {text!r}")
    return key, value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--policy", required=True, help="The ServedPolicy subclass, module:Class (importable).")
    parser.add_argument("--checkpoint", help="Passed to the policy's load(); the launcher's {checkpoint}.")
    parser.add_argument("--port", type=int, required=True, help="The launcher's {port}.")
    parser.add_argument("--host", default="127.0.0.1", help="The worker reaches its server on 127.0.0.1.")
    parser.add_argument("--token-file", type=Path,
                        help="Refuse requests without this bearer token; the launcher's {token_file}. Without it, any "
                             "request is answered (a local check only).")
    parser.add_argument("--option", type=_option, action="append", default=[], metavar="KEY=VALUE",
                        help="Passed to the policy's load() as options[KEY] = VALUE (strings); repeat for more.")
    args = parser.parse_args(argv)
    token = args.token_file.read_text().strip() if args.token_file else None
    if args.token_file and not token:
        parser.error(f"{args.token_file} is empty")
    try:
        serve(load_policy(args.policy), args.checkpoint, args.host, args.port, token, dict(args.option))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
