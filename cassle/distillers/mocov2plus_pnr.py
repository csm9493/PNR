import argparse
from typing import Any, List, Sequence

import torch
from torch import nn
from cassle.distillers.base import base_distill_wrapper
from cassle.losses.pnr import moco_pnr_loss_func
from cassle.utils.gather_layer import gather
import torch.nn.functional as F


def mocov2plus_pnr_wrapper(Method=object):
    class Mocov2plus_PNR_Wrapper(base_distill_wrapper(Method)):
        def __init__(
            self,
            distill_lamb: float,
            distill_proj_hidden_dim: int,
            distill_temperature: float,
            loss_alpha: float,
            **kwargs
        ):
            super().__init__(**kwargs)

            self.distill_lamb = distill_lamb
            self.distill_temperature = distill_temperature
            self.loss_alpha = loss_alpha
            print (' self.distill_temperature : ', self.distill_temperature)
            output_dim = kwargs["output_dim"]

            self.distill_predictor = nn.Sequential(
                nn.Linear(output_dim, distill_proj_hidden_dim),
                nn.BatchNorm1d(distill_proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(distill_proj_hidden_dim, output_dim),
            )


            self.register_buffer("queue2", torch.randn(2, output_dim, self.queue_size))
            self.queue2 = nn.functional.normalize(self.queue2, dim=1)
            self.register_buffer("queue2_ptr", torch.zeros(1, dtype=torch.long))
            

        @staticmethod
        def add_model_specific_args(
            parent_parser: argparse.ArgumentParser,
        ) -> argparse.ArgumentParser:
            parser = parent_parser.add_argument_group("contrastive_distiller")

            parser.add_argument("--distill_lamb", type=float, default=1)
            parser.add_argument("--distill_proj_hidden_dim", type=int, default=2048)
            parser.add_argument("--distill_temperature", type=float, default=0.2)
            parser.add_argument("--loss_alpha", type=float, default=0.5)

            return parent_parser
        
        def _dequeue_and_enqueue2(self, keys: torch.Tensor):
            """Adds new samples and removes old samples from the queue in a fifo manner.

            Args:
                keys (torch.Tensor): output features of the momentum encoder.
            """

            batch_size = keys.shape[1]
            ptr = int(self.queue2_ptr)  # type: ignore
            assert self.queue_size % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            keys = keys.permute(0, 2, 1)
            self.queue2[:, :, ptr : ptr + batch_size] = keys
            ptr = (ptr + batch_size) % self.queue_size  # move pointer
            self.queue2_ptr[0] = ptr  # type: ignore

        @property
        def learnable_params(self) -> List[dict]:
            """Adds distill predictor parameters to the parent's learnable parameters.

            Returns:
                List[dict]: list of learnable parameters.
            """

            extra_learnable_params = [
                {"params": self.distill_predictor.parameters()},
            ]
            return super().learnable_params + extra_learnable_params

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
            out = super().training_step(batch, batch_idx)
            
            z1, z2 = out["z"]
            k1, k2 = out["k"]
            
            frozen_z1, frozen_z2 = out["frozen_z"]
            frozen_z1 = F.normalize(frozen_z1, dim=-1)
            frozen_z2 = F.normalize(frozen_z2, dim=-1)

            p1 = self.distill_predictor(z1)
            p2 = self.distill_predictor(z2)
            p1 = F.normalize(p1, dim=-1)
            p2 = F.normalize(p2, dim=-1)
            
            queue1 = out["queue"]
            queue2 = self.queue2.clone().detach()

            # ------- update queue -------
            keys2 = torch.stack((gather(frozen_z1), gather(frozen_z2)))
            self._dequeue_and_enqueue2(keys2)
            
            loss = (
                moco_pnr_loss_func(z1, k2, queue1[1], p1, frozen_z1, k1, queue2[1], p2, self.temperature, self.distill_temperature, self.loss_alpha)
                + moco_pnr_loss_func(z2, k1, queue2[0], p2, frozen_z2, k2, queue1[0], p1, self.temperature, self.distill_temperature, self.loss_alpha)
            ) / 2

            self.log("train_moco_pnr_loss", loss, on_epoch=True, sync_dist=True)

            out.update({"loss": out["loss"] + loss})
            
            return out

    return Mocov2plus_PNR_Wrapper

