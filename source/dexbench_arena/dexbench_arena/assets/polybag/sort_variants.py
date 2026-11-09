# SPDX-License-Identifier: Apache-2.0
"""DB-L1's polybag pool: eleven bag bodies (film colour x size) times three waybills, 33 packages."""

from __future__ import annotations

from dataclasses import dataclass

BASE_PACKAGE = "polybag_cotton_tet_deformable"
VARIANT_PACKAGE = "polybag_tet_sort_variants"
BASE_FOOTPRINT_M = (0.249, 0.3215)
BASE_NOMINAL_CM = (25, 35)


@dataclass(frozen=True)
class BagBody:
    key: str
    colour_name: str
    """Korean catalogue name of the film colour."""
    rgb: tuple[int, int, int]
    """sRGB film colour."""
    nominal_cm: tuple[int, int]
    """Courier-bag size, width x length (the +4 cm flap not counted)."""
    split: str
    """GA-O split: "train", "eval_colour" (held out: a colour no training body has) or "eval_shape" (held out: an
    outline no training body has)."""
    roughness: float = 0.35
    metallic: float = 0.0

    @property
    def scale_xy(self) -> tuple[float, float]:
        return (self.nominal_cm[0] / BASE_NOMINAL_CM[0], self.nominal_cm[1] / BASE_NOMINAL_CM[1])

    @property
    def footprint_m(self) -> tuple[float, float]:
        sx, sy = self.scale_xy
        return (BASE_FOOTPRINT_M[0] * sx, BASE_FOOTPRINT_M[1] * sy)


BAG_BODIES: tuple[BagBody, ...] = (
    BagBody("grey_25x35", "진회색", (92, 94, 99), (25, 35), "train"),
    BagBody("black_25x35", "검정", (32, 32, 35), (25, 35), "train"),
    BagBody("white_25x35", "화이트(유백)", (232, 232, 226), (25, 35), "train", roughness=0.5),
    BagBody("pink_20x30", "핫핑크", (222, 72, 142), (20, 30), "train"),
    BagBody("silver_30x40", "실버", (186, 189, 194), (30, 40), "train", roughness=0.25, metallic=0.35),
    BagBody("grey_20x30", "진회색", (92, 94, 99), (20, 30), "train"),
    BagBody("black_30x40", "검정", (32, 32, 35), (30, 40), "train"),
    BagBody("purple_25x35", "보라", (118, 82, 160), (25, 35), "eval_colour"),
    BagBody("skyblue_20x30", "소라(하늘)", (146, 196, 226), (20, 30), "eval_colour"),
    BagBody("grey_35x45", "진회색", (92, 94, 99), (35, 45), "eval_shape"),
    BagBody("black_20x40", "검정", (32, 32, 35), (20, 40), "eval_shape"),
)


@dataclass(frozen=True)
class WaybillClass:
    tote: int
    """The sort tote 1-3 for this SUB code; DB-L1 (one tote) does not read it."""
    sub_code: str
    """SUB sort code, printed largest."""
    route: str
    """Delivery route (driver) printed under the sort code."""
    street: str
    """Recipient's street: the three routes split one fictional district (romanised)."""
    postcode_prefix: str


DESTINATION_CODE = "5K21"
"""The destination (sub-terminal) code every bag shares: one station, one sub-terminal."""
SUB_TERMINAL = "SEONGSU SUB"
CARRIER = "DX EXPRESS"

WAYBILL_CLASSES: tuple[WaybillClass, ...] = (
    WaybillClass(1, "1A", "SEONGSU A-07", "Dexseongsu 1ga-gil", "0478"),
    WaybillClass(2, "2B", "SEONGSU B-12", "Dexseongsu 2ga-gil", "0479"),
    WaybillClass(3, "3C", "SONGJEONG C-03", "Dexsongjeong-gil", "0480"),
)


THICKNESS_LEVELS_M = (0.08, 0.076, 0.072)
YOUNGS_MODULI_PA = (2_500.0,)


@dataclass(frozen=True)
class SortVariant:
    body: BagBody
    waybill: WaybillClass
    thickness_m: float
    """The bag's thickness in the middle [m] (bottom to top, the pillow's full height)."""
    youngs_modulus: float
    """Its core tets' Young's modulus [Pa], the bound physics material's."""

    @property
    def key(self) -> str:
        return f"{self.body.key}__{self.waybill.sub_code.lower()}"

    @property
    def sort_code(self) -> str:
        return f"{DESTINATION_CODE}-{self.waybill.sub_code}"

    @property
    def tote(self) -> int:
        return self.waybill.tote

    @property
    def split(self) -> str:
        return self.body.split


SORT_VARIANTS: tuple[SortVariant, ...] = tuple(
    SortVariant(b, w, THICKNESS_LEVELS_M[j], YOUNGS_MODULI_PA[(i + j) % len(YOUNGS_MODULI_PA)])
    for i, b in enumerate(BAG_BODIES) for j, w in enumerate(WAYBILL_CLASSES))
SORT_VARIANTS_BY_KEY = {v.key: v for v in SORT_VARIANTS}


def variant_usd_relpath(key: str) -> str:
    """The variant's USD, relative to ``assets/object_library`` (a layer the package's main file selects)."""
    return f"{VARIANT_PACKAGE}/usd/variants/{key}.usd"


BOX_VARIANT_PACKAGE = "box_krpost_sort_variants"


@dataclass(frozen=True)
class BoxVariant:
    number: int
    """Korea Post box number (1-3): package ``box_krpost_fc_0<number>``."""
    size_m: tuple[float, float, float]
    """Bounding box at scale 1 (origin on the bottom face, centred in XY)."""
    waybill: WaybillClass
    copy: int
    """Which of the box's labelled copies (labels differ by copy)."""

    @property
    def key(self) -> str:
        return f"krpost{self.number}_{self.copy}__{self.waybill.sub_code.lower()}"

    @property
    def base_package(self) -> str:
        return f"box_krpost_fc_0{self.number}"

    @property
    def asset(self) -> str:
        """The registered Arena asset (a dynamic rigid body)."""
        return f"dexbench_l1_box_{self.key}"

    @property
    def sort_code(self) -> str:
        return f"{DESTINATION_CODE}-{self.waybill.sub_code}"

    @property
    def dim_text(self) -> str:
        return "BOX {:.0f}x{:.0f}x{:.0f} CM".format(*(100 * v for v in self.size_m))


_KRPOST_SIZE_M = {1: (0.221, 0.190, 0.091), 2: (0.271, 0.180, 0.151), 3: (0.341, 0.250, 0.211)}
_W = {w.sub_code: w for w in WAYBILL_CLASSES}
BOX_VARIANTS: tuple[BoxVariant, ...] = (
    BoxVariant(1, _KRPOST_SIZE_M[1], _W["1A"], 1),
    BoxVariant(1, _KRPOST_SIZE_M[1], _W["3C"], 2),
    BoxVariant(2, _KRPOST_SIZE_M[2], _W["2B"], 1),
    BoxVariant(2, _KRPOST_SIZE_M[2], _W["1A"], 2),
    BoxVariant(3, _KRPOST_SIZE_M[3], _W["3C"], 1),
    BoxVariant(3, _KRPOST_SIZE_M[3], _W["2B"], 2),
)
BOX_VARIANTS_BY_KEY = {v.key: v for v in BOX_VARIANTS}


def box_variant_usd_relpath(key: str) -> str:
    """The box variant's USD, relative to ``assets/object_library`` (a layer the package's main file selects)."""
    return f"{BOX_VARIANT_PACKAGE}/usd/variants/{key}.usd"

