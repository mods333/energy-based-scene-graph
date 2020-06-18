from .model_ebm import GraphEnergyModel

_ENERGY_META_ARCHITECTURES = {"GraphEnergyModel": GraphEnergyModel}

def build_energy_model(cfg, obj_classes, rel_classes, in_channels):
    meta_arch = _ENERGY_META_ARCHITECTURES[cfg.ENERGY_MODEL.META_ARCHITECTURE]
    return meta_arch(cfg, obj_classes, rel_classes, in_channels)