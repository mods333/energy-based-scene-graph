# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT
from .model_ebm import GraphEnergyModel

_ENERGY_META_ARCHITECTURES = {"GraphEnergyModel": GraphEnergyModel}

def build_energy_model(cfg, in_channels):
    meta_arch = _ENERGY_META_ARCHITECTURES[cfg.ENERGY_MODEL.META_ARCHITECTURE]
    return meta_arch(cfg, in_channels)