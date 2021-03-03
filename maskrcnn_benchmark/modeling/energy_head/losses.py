# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT
import torch
from maskrcnn_benchmark.modeling import registry

@registry.ENERGY_LOSS.register("ContrastiveDivergence")
def ContrastiveDivergence(cfg, positive_energy, negative_energy):

    pos_loss = torch.mean(cfg.ENERGY_MODEL.TEMP * positive_energy)
    neg_loss = torch.mean(cfg.ENERGY_MODEL.TEMP * negative_energy)
    loss_ml =  pos_loss - neg_loss + torch.sum(positive_energy**2 + negative_energy**2)

    return {'ML Loss (cd)': loss_ml}

@registry.ENERGY_LOSS.register("SoftPlus")
def SoftPlus(cfg, positive_energy, negative_energy):

    loss_ml = torch.nn.Softplus()(cfg.ENERGY_MODEL.TEMP*(positive_energy- negative_energy))
    return {'ML Loss (sp)': loss_ml}

@registry.ENERGY_LOSS.register("LogSumExp")
def LogSumExp(cfg, positive_energy, negative_energy):

    negative_energy_reduced = (negative_energy - torch.min(negative_energy))

    coeff = torch.exp(-cfg.ENERGY_MODEL.TEMP*negative_energy_reduced)
    norm_const = torch.sum(coeff) + 1e-4

    pos_term = cfg.ENERGY_MODEL.TEMP* positive_energy
    pos_loss = torch.mean(pos_term)

    neg_loss = coeff * (-cfg.ENERGY_MODEL.TEMP*negative_energy_reduced) / norm_const

    loss_ml = pos_loss + torch.sum(neg_loss)

    return {'ML Loss (lse)': loss_ml}

def build_loss_function(cfg):
    loss_func = registry.ENERGY_LOSS[cfg.ENERGY_MODEL.LOSS]

    return loss_func