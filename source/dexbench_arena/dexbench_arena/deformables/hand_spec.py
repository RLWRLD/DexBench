# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""What the paper sidecar needs to know about a robot hand to mirror it into the paper world."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class PaperHandSpec:
    """Names of one robot hand, for the paper sidecar's hand mirror."""

    palm_link: str
    """The hand's base body."""
    hull_link_pattern: str
    """Regex (anchored by :meth:`hull_links`) of the links whose collision hulls
    the paper sees. A fixed link that only duplicates its parent is left out."""

    @staticmethod
    def fmt(template: str, side: str) -> str:
        return template.format(side=side)

    def hull_links(self, body_names: list[str], side: str) -> list[str]:
        """Links whose hulls the paper sees, in body order."""

        pattern = re.compile(rf"^({self.hull_link_pattern.format(side=side)})$")
        return [n for n in body_names if pattern.match(n)]
