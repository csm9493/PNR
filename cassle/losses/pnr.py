import torch
import torch.nn.functional as F

import torch.distributed as dist

def moco_pnr_loss_func(
    query1: torch.Tensor, key1: torch.Tensor, queue1: torch.Tensor, distill1: torch.Tensor, query2: torch.Tensor, key2: torch.Tensor, queue2: torch.Tensor, distill2: torch.Tensor, temperature=0.1, distill_temperature=0.1, loss_alpha = 0.5
) -> torch.Tensor:
    """Computes MoCo's loss given a batch of queries from view 1, a batch of keys from view 2 and a
    queue of past elements.

    Args:
        query (torch.Tensor): NxD Tensor containing the queries from view 1.
        key (torch.Tensor): NxD Tensor containing the queries from view 2.
        queue (torch.Tensor): a queue of negative samples for the contrastive loss.
        temperature (float, optional): [description]. temperature of the softmax in the contrastive
            loss. Defaults to 0.1.

    Returns:
        torch.Tensor: MoCo loss.
    """


    l_pos1 = torch.einsum('nc,nc->n', [query1, key1]).unsqueeze(-1)
    l_neg1 = torch.einsum('nc,ck->nk', [query1, torch.cat([queue1, queue2], dim = 1)])

    # logits: Nx(1+K)
    logits1 = torch.cat([l_pos1, l_neg1], dim=1)

    # apply temperature
    logits1 /= temperature

    # labels: positive key indicators
    labels = torch.zeros(logits1.shape[0], dtype=torch.long).cuda()

    loss1 = F.cross_entropy(logits1, labels)

    l_pos2 = torch.einsum('nc,nc->n', [distill1, query2]).unsqueeze(-1)
    l_neg2 = torch.einsum('nc,ck->nk', [distill1, torch.cat([queue1, queue2], dim = 1)])
    
    # logits: Nx(1+K)
    logits2 = torch.cat([l_pos2, l_neg2], dim=1)

    # apply temperature
    logits2 /= distill_temperature

    loss2 = F.cross_entropy(logits2, labels)
    
    loss = loss1 + loss2
    
    return loss


def simclr_moco_loss_func(
    p1: torch.Tensor,
    p2: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    f1: torch.Tensor,
    f2: torch.Tensor,
    temperature: float = 0.1,
    labels: torch.Tensor = None,
) -> torch.Tensor:
    
    device = z1.device

    b = z1.size(0)
#     z = torch.cat((z1, z2), dim=0)
#     z = F.normalize(z, dim=-1)

    p = F.normalize(torch.cat([p1, p2], dim=0), dim=-1)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)
    f = F.normalize(torch.cat([f1, f2], dim=0), dim=-1)

    logits1 = torch.einsum("if, jf -> ij", z, z) / temperature
    logits1_prev = torch.einsum("if, jf -> ij", z, f) / temperature
    
    logits2 = torch.einsum("if, jf -> ij", p, f) / temperature
    logits2_prev = torch.einsum("if, jf -> ij", p, z) / temperature
    
    logits1_cat = torch.cat((logits1, logits1_prev), dim = 1)
    logits2_cat = torch.cat((logits2, logits2_prev), dim = 1)
    
    
    logits1_max, _ = torch.max(logits1_cat, dim=1, keepdim=True)
    logits1_cat = logits1_cat - logits1_max.detach()
    
    logits2_max, _ = torch.max(logits2_cat, dim=1, keepdim=True)
    logits2_cat = logits2_cat - logits2_max.detach()
    
    pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
    pos_mask[:, b:].fill_diagonal_(True)
    pos_mask[b:, :].fill_diagonal_(True)
    
    pos_mask2 = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
    pos_mask2.fill_diagonal_(True)
    
    # all matches excluding the main diagonal
    logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)
    logit_mask_cat = torch.cat((logit_mask, logit_mask), dim = 1)

#     print (logits1_cat.shape, logit_mask_cat.shape)
    exp_logits1 = torch.exp(logits1_cat) * logit_mask_cat
    log_prob1 = logits1_cat - torch.log(exp_logits1.sum(1, keepdim=True))
    
    exp_logits2 = torch.exp(logits2_cat) * logit_mask_cat
    log_prob2 = logits2_cat - torch.log(exp_logits2.sum(1, keepdim=True))
    
    zeros_mask = torch.zeros_like(pos_mask)
    pos_mask = torch.cat((pos_mask, zeros_mask), dim = 1)
    pos_mask2 = torch.cat((pos_mask2, zeros_mask), dim = 1)

    # compute mean of log-likelihood over positives
    mean_log_prob_pos1 = (pos_mask * log_prob1).sum(1) / pos_mask.sum(1)
    # loss
    loss1 = -mean_log_prob_pos1.mean()
    
    # compute mean of log-likelihood over positives
    mean_log_prob_pos2 = (pos_mask2 * log_prob2).sum(1) / pos_mask.sum(1)
    # loss
    loss2 = -mean_log_prob_pos2.mean()
    
    loss = loss1 + loss2

    return loss



