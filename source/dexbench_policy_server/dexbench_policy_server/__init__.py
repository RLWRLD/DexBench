"""Serve a policy to the DexBench benchmark: subclass :class:`ServedPolicy` (``load``, ``info``, ``predict``) and run"""

from dexbench_policy_server.policy import ServedPolicy
from dexbench_policy_server.server import CONTRACT, READY_LINE, VERSION, make_server, serve

__version__ = VERSION
__all__ = ["CONTRACT", "READY_LINE", "VERSION", "ServedPolicy", "__version__", "make_server", "serve"]
