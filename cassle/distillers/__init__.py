from cassle.distillers.base import base_distill_wrapper
from cassle.distillers.contrastive import contrastive_distill_wrapper
from cassle.distillers.decorrelative import decorrelative_distill_wrapper
from cassle.distillers.predictive import predictive_distill_wrapper
from cassle.distillers.predictive_mse import predictive_mse_distill_wrapper

from cassle.distillers.barlow_pnr import barlow_pnr_wrapper
from cassle.distillers.byol_pnr import byol_pnr_wrapper
from cassle.distillers.mocov2plus_pnr import mocov2plus_pnr_wrapper
from cassle.distillers.simclr_pnr import simclr_pnr_wrapper
from cassle.distillers.vicreg_pnr import vicreg_pnr_wrapper

__all__ = [
    "base_distill_wrapper",
    "contrastive_distill_wrapper",
    "decorrelative_distill_wrapper",
    "predictive_distill_wrapper",
    "predictive_mse_distill_wrapper",
    "barlow_pnr_wrapper",
    "byol_pnr_wrapper",
    "mocov2plus_pnr_wrapper",
    "simclr_pnr_wrapper",
    "vicreg_pnr_wrapper",
]

DISTILLERS = {
    "base": base_distill_wrapper,
    "contrastive": contrastive_distill_wrapper,
    "decorrelative": decorrelative_distill_wrapper,
    "predictive": predictive_distill_wrapper,
    "predictive_mse": predictive_mse_distill_wrapper,
    "barlow_pnr": barlow_pnr_wrapper,
    "byol_pnr": byol_pnr_wrapper,
    "mocov2plus_pnr": mocov2plus_pnr_wrapper,
    "simclr_pnr": simclr_pnr_wrapper,
    "vicreg_pnr": vicreg_pnr_wrapper,
}
