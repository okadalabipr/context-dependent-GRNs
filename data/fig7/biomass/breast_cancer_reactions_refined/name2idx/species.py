from dataclasses import make_dataclass
from typing import Dict, List

NAMES: List[str] = [
    "RAS_inact",
    "RAS_act",
    "PI3K_inact",
    "PI3K_act",
    "RAF1_inact",
    "RAF1_act",
    "AKT_inact",
    "AKT_p",
    "Heregulin",
    "HER3",
    "Heregulin_HER3",
    "HER2",
    "Heregulin_HER3_HER2",
    "Heregulin_HER3_HER2_p",
    "EGF",
    "EGFR",
    "EGF_EGFR",
    "EGF_EGFR_EGF_EGFR",
    "EGF_EGFR_p",
    "EGFR_EGFR",
    "EGFR_EGFR_p",
    "EGFR_HER2",
    "EGFR_HER2_p",
    "Shc",
    "Shc_p",
    "MEK",
    "MEK_p",
    "ERK",
    "ERK_p_cytoplasm",
    "ERK_p_nucleus",
    "RSK",
    "RSK_p",
    "cFOS",
    "cFOS_p",
    "PIP2",
    "PIP3",
    "PTEN",
    "GSK3B",
    "GSK3B_p",
    "CDKN1A",
    "DUSP",
    "SPRY",
    "cFOS_mRNA",
]

NUM: int = len(NAMES)

Species = make_dataclass(
    cls_name="Species",
    fields=[(name, int) for name in NAMES],
    namespace={"NAMES": NAMES, "NUM": NUM},
    frozen=True,
)

name2idx: Dict[str, int] = {k: v for v, k in enumerate(NAMES)}

V = Species(**name2idx)

del name2idx
