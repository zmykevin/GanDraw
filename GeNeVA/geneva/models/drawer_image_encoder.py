import torch
import torch.nn as nn
from torchvision.models import densenet161, resnet152, vgg19

# Encoder


class DrawerImageEncoder(nn.Module):

    def __init__(self, cfg):
        super(DrawerImageEncoder, self).__init__()
        self.cfg = cfg
        network = self.cfg.teller_img_encoder_net
        self.pretrained_image_network = network
        if cfg.drawer_dialog_img_input:
            self.net = resnet152(pretrained=True)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        else:
            self.net = nn.Sequential(
                nn.Conv2d(22, 64, 4, 2, 1, bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, cfg.image_feat_dim, 4, 2, 1,
                          bias=False),
                nn.BatchNorm2d(cfg.image_feat_dim),
            )
            self.dim = self.cfg.image_feat_dim
        cfg.img_encoder_dim = self.dim

    def forward(self, x):
        x = self.net(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return x