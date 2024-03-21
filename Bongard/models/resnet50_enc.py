# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Bongard-HOI. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import nn, Tensor
# from torchvision.ops import roi_align
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
import models
import utils
from .models import register

import torchvision.ops.boxes as box_ops
from typing import List, Tuple
from torchvision.transforms import Resize

import random
import pdb
from random import shuffle
@register('resnet50_enc')
class RelationnetBBoxNetworkEncoder(nn.Module):
    def __init__(self, encoder, **kwargs):
        super(RelationnetBBoxNetworkEncoder, self).__init__()

        # image encoder
        encoder = models.make(encoder)
        self.encoder = encoder
        self.proj = nn.Conv2d(encoder.out_dim, encoder.out_dim // 2, kernel_size=1)


    def forward(self, im, shot_box, shot_hum, roi_pos):
        img_shape = im.shape
        if len(img_shape) < 6:
            img_shape = im.unsqueeze(dim = 2).shape


        im = im.view(-1, *img_shape[-3:])

                          
        x = self.encoder(im, roi_position=False)

        if len(x.shape) != 4:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.proj(x)
        x = x.squeeze().view(img_shape[0]*img_shape[1], img_shape[2], -1)

        return x


if __name__ == '__main__':
    im = torch.rand((8, 3, 128, 128))

    model = RelationnetBBoxNetworkEncoder(encoder='resnet50')
    x = model(im)
    print(x.shape)
