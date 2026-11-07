"""LFPformer release: retain the pinned wav2vec2 implementation only."""
from backbones.base import BackboneBase
from backbones.wav2vec2_backbone import Wav2Vec2Backbone
BACKBONE_REGISTRY = {"wav2vec2": Wav2Vec2Backbone}
def get_backbone(name, config):
    if name not in BACKBONE_REGISTRY:
        raise ValueError("This release packages wav2vec2; other experimental backbones are not supported")
    return BACKBONE_REGISTRY[name](config)
