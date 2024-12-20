import argparse
from typing import Any, List, Sequence

import torch
from torch import nn
from cassle.distillers.base import base_distill_wrapper
from cassle.losses.byol import byol_loss_func


def byol_pnr_wrapper(Method=object):
    class BYOL_PNR_Wrapper(base_distill_wrapper(Method)):
        def __init__(self, distill_lamb: float, distill_proj_hidden_dim, **kwargs):
            super().__init__(**kwargs)

            self.distill_lamb = distill_lamb
            output_dim = kwargs["output_dim"]

            self.distill_predictor = nn.Sequential(
                nn.Linear(output_dim, distill_proj_hidden_dim),
                nn.BatchNorm1d(distill_proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(distill_proj_hidden_dim, output_dim),
            )
            
            print ('self.negative_lamb : ', self.negative_lamb)

        @staticmethod
        def add_model_specific_args(
            parent_parser: argparse.ArgumentParser,
        ) -> argparse.ArgumentParser:
            parser = parent_parser.add_argument_group("contrastive_distiller")

            parser.add_argument("--distill_lamb", type=float, default=1)
            parser.add_argument("--distill_proj_hidden_dim", type=int, default=2048)

            return parent_parser

        @property
        def learnable_params(self) -> List[dict]:
            """Adds distill predictor parameters to the parent's learnable parameters.

            Returns:
                List[dict]: list of learnable parameters.
            """

            extra_learnable_params = [
                {
                    "name": "distill_predictor",
                    "params": self.distill_predictor.parameters(),
                    "lr": self.lr if self.distill_lamb >= 1 else self.lr / self.distill_lamb,
                    "weight_decay": self.weight_decay,
                },
            ]
            return super().learnable_params + extra_learnable_params

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
            out = super().training_step(batch, batch_idx)
            z1, z2 = out["z"]
            frozen_z1, frozen_z2 = out["frozen_z"]

            p1 = self.distill_predictor(z1)
            p2 = self.distill_predictor(z2)

            positive_loss = (byol_loss_func(p1, frozen_z1) + byol_loss_func(p2, frozen_z2)) / 2
            negative_loss = (byol_loss_func(p1, frozen_z2) + byol_loss_func(p2, frozen_z1)) / 2

            self.log("train_byol_pnr_positive_loss", positive_loss, on_epoch=True, sync_dist=True)
            self.log("train_byol_pnr_negative_loss", negative_loss, on_epoch=True, sync_dist=True)

            return out["loss"] + self.distill_lamb * positive_loss - self.negative_lamb * negative_loss

    return BYOL_PNR_Wrapper
