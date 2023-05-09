import torch
from torch import nn

from models.utils import SpikingBlock, get_model
from models.SSD_utils import init_weights

class DetectionBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
       
        self.model = get_model(args)
       
        if args.pretrained_backbone is not None:
            ckpt = torch.load(args.pretrained_backbone)
            state_dict = {k.replace('model.', ''):v for k,v in ckpt['state_dict'].items()}
            self.model.load_state_dict(state_dict, strict=False)
            
        self.out_channels = self.model.out_channels
        extras_fm = args.extras
        
        self.extras = nn.ModuleList(
            [
                nn.Sequential(
                    SpikingBlock(self.out_channels[-1], self.out_channels[-1]//2, kernel_size=1),
                    SpikingBlock(self.out_channels[-1]//2, extras_fm[0], kernel_size=3, padding=1, stride=2),
                ),
                nn.Sequential(
                    SpikingBlock(extras_fm[0], extras_fm[0]//4, kernel_size=1),
                    SpikingBlock(extras_fm[0]//4, extras_fm[1], kernel_size=3, padding=1, stride=2),
                ),
                nn.Sequential(
                    SpikingBlock(extras_fm[1], extras_fm[1]//2, kernel_size=1),
                    SpikingBlock(extras_fm[1]//2, extras_fm[2], kernel_size=3, padding=1, stride=2),
                ),
            ]
        )

        self.extras.apply(init_weights)
        self.out_channels.extend(extras_fm)
    
    def forward(self, x):
        # print(x.shape)
        feature_maps = self.model(x, classify=False)
        x = feature_maps[-1]
        detection_feed = [fm.sum(dim=1) for fm in feature_maps]

        for block in self.extras:
            x = block(x)
            detection_feed.append(x.sum(dim=1))
            
        return detection_feed