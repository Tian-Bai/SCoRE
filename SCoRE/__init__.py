"""Public API for SCoRE."""

from .core import cost_ebh, score_evalues

from .SCoRE import (
    CS,
    SCoRE_MDR,
    SCoRE_MDR_bf,
    SCoRE_MDR_w,
    SCoRE_MSR,
    SCoRE_MSR_bf,
    SCoRE_MSR_w,
    SCoRE_SDR,
    SCoRE_SDR_w,
    SCoRE_WSR,
    SCoRE_WSR_w,
)
from .utility import (
    BH,
    Lpredictor,
    eBH,
    eval_MDR,
    eval_MSR,
    eval_SDR,
    eval_WSR,
    gen_data_1,
    gen_data_2,
    gen_data_Jin2023,
    loss_1,
    loss_2,
    loss_Jin2023,
)

__version__ = "0.1.1"

__all__ = [
    "__version__",
    "BH",
    "CS",
    "Lpredictor",
    "SCoRE_MDR",
    "SCoRE_MDR_bf",
    "SCoRE_MDR_w",
    "SCoRE_MSR",
    "SCoRE_MSR_bf",
    "SCoRE_MSR_w",
    "SCoRE_SDR",
    "SCoRE_SDR_w",
    "SCoRE_WSR",
    "SCoRE_WSR_w",
    "eBH",
    "cost_ebh",
    "score_evalues",
    "eval_MDR",
    "eval_MSR",
    "eval_SDR",
    "eval_WSR",
    "gen_data_1",
    "gen_data_2",
    "gen_data_Jin2023",
    "loss_1",
    "loss_2",
    "loss_Jin2023",
]
