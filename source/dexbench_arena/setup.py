"""Installation script for the ``dexbench_arena`` IsaacLab-Arena extension."""

from __future__ import annotations

import os

import toml
from setuptools import find_packages, setup


EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

INSTALL_REQUIRES = [
    "huggingface_hub>=0.23",
    "onnxscript",
    "psutil",
    "wandb>=0.19",
]


setup(
    name="dexbench-arena",
    packages=find_packages(include=["dexbench_arena*"]),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="Apache-2.0",
    include_package_data=True,
    package_data={
        "dexbench_arena": [
            "embodiments/rby1_wuji2/data/dex_retargeting/*.yml",
            "assets/e4_hotel_fb_dishes.json",
            "environments/manipulation/db_m1_grasp_plan_v1.json",
            "benchmark/frozen/*.json",
        ],
    },
    python_requires=">=3.11",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Isaac Sim :: 6.1.0",
    ],
    zip_safe=False,
)
