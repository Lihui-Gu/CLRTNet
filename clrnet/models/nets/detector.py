import torch.nn as nn
import torch

from clrnet.models.registry import NETS
from ..registry import build_backbones, build_aggregator, build_heads, build_necks, build_timesformer


@NETS.register_module
class Detector(nn.Module):
    def __init__(self, cfg):
        super(Detector, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbones(cfg)
        self.aggregator = build_aggregator(cfg) if cfg.haskey('aggregator') else None
        self.neck = build_necks(cfg) if cfg.haskey('neck') else None
        self.heads = build_heads(cfg)
        self.timesformer = build_timesformer(cfg) if cfg.haskey('timesformer') else None

    def get_lanes(self):
        return self.heads.get_lanes(output)

    def forward(self, batch):
        output = {}
        # fea 是backbone输出的list类型
        fea_backbone = self.backbone(batch['img'] if isinstance(batch, dict) else batch)
        if self.neck:
            fea = self.neck(fea_backbone)
        else:
            fea = self.timesformer(fea_backbone)
        if self.training:
            output = self.heads(fea, batch=batch)
        else:
            output = self.heads(fea)
        return output
