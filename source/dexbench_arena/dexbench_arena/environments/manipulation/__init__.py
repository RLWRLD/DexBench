# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DexBench manipulation environment registrations."""

from .e4_hotel_fb_environment import DexBenchHotelFBE4Environment
from .e4_hotel_fb_pour_environment import DexBenchHotelFBE4PourEnvironment
from .e4_h3_environment import H3Environment
from .e2_logistics_environment import DexBenchLogisticsCenterE2Environment
from .e2_logistics_l1_environment import DexBenchLogisticsE2L1Environment
from .e2_logistics_l4_environment import DexBenchLogisticsE2L4Environment
from .e3_assembly_line_environment import (
    DexBenchManufacturingE3Environment,
    DexBenchManufacturingE3M1Environment,
    DexBenchManufacturingE3M4Environment,
    DexBenchManufacturingM5MovingConveyorKittingEnvironment,
)
from .e3_manufacturing_db_m2_environment import DexBenchManufacturingE3M2Environment
from .e1_retail_environment import DexBenchRetailStoreE1Environment

__all__ = [
    "DexBenchHotelFBE4Environment",
    "DexBenchHotelFBE4PourEnvironment",
    "H3Environment",
    "DexBenchLogisticsCenterE2Environment",
    "DexBenchLogisticsE2L1Environment",
    "DexBenchLogisticsE2L4Environment",
    "DexBenchManufacturingE3Environment",
    "DexBenchManufacturingE3M1Environment",
    "DexBenchManufacturingE3M2Environment",
    "DexBenchManufacturingE3M4Environment",
    "DexBenchManufacturingM5MovingConveyorKittingEnvironment",
    "DexBenchRetailStoreE1Environment",
]
