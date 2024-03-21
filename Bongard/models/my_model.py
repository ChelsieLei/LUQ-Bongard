# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Bongard-HOI. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models
import utils
from .models import register
from .subspace_projection import Subspace_Projection
from .SupCtsloss import SupConLoss
import scipy.stats
import pdb


@register('my_model')
class My_model(nn.Module):

    def __init__(self, encoder, encoder_args={}, method = 'cosine', type = 'raw'):
        super().__init__()
        print('encoder: {}'.format(encoder))
        print('encoder_args: {}'.format(encoder_args))
        self.encoder = models.make(encoder, **encoder_args)
        self.flag = 0
        if type == 'raw':  ##use f to build subspace
            self.projection_pro = Subspace_Projection(num_dim=5, type=method)  ### subspace dimension
        elif type == 'dsn':   ## use f-mu to build subspaces
            self.projection_pro = Subspace_Projection_DSN(num_dim=5, type=method)
        if method == 'cosine':
            self.temp = nn.Parameter(torch.tensor(10.))
        elif method == 'l2_dist':
            self.temp = nn.Parameter(torch.tensor(0.1))
        self.cts_loss_func = SupConLoss()
        # self.classifier = Classifier(1280, 2) ### for test time adaptation

    def forward(self, x_shot, x_query, **kwargs):  ## flag 0 normal subspace; flag 1 positive bias subspace

        shot_shape = x_shot.shape[:-3]
        img_shape = x_shot.shape[-3:]

        assert 'query_boxes' in kwargs
        # if 'ema' in kwargs.keys() and kwargs['ema'] == True:
        #     # print(kwargs['teach_neg_ims'].shape)
        #     # x_shot =  torch.stack((x_shot[:,0], kwargs['teach_neg_ims']), dim=1)
        #     x_shot =  torch.cat((x_shot[:,0].unsqueeze(1), kwargs['teach_neg_ims'].view(2, -1, *img_shape).unsqueeze(0)), dim=1)
        #     # print("x_shot: ",x_shot.shape)
        #     x_shot = self.encoder(x_shot, [kwargs['shot_boxes'][i][:6] + kwargs['teach_neg_boxes'][i] for i in range(shot_shape[0])], 
        #                           [kwargs['shot_human_roi'][i][:6] + kwargs['teach_neg_hum_roi'][i] for i in range(shot_shape[0])], kwargs['roi_position'])
        # else:    
        x_shot = self.encoder(x_shot, kwargs['shot_boxes'], kwargs['shot_human_roi'], kwargs['roi_position'])  ##[pos_i, neg_i, pos_(i+1), neg_(i+1), ...]
        if isinstance(x_shot, list):
            x_shot = torch.stack(x_shot)    
        

        if 'query_boxes' in kwargs:
            assert 'shot_boxes' in kwargs
            x_query = self.encoder(x_query, kwargs['query_boxes'], kwargs['query_human_roi'],  kwargs['roi_position'])
        else:
            x_query = x_query.view(-1, *img_shape)
            x_query = self.encoder(x_query)
        if isinstance(x_query,list):
            x_query = torch.stack(x_query)   
        # else:
        #     x_query = x_query.unsqueeze(0)

        logits, dist_loss, cts_loss = [], [], []
        pos_set, neg_set = [], []
        for i in range(shot_shape[0]):   ### process each instance in the batch seperately
            # if 'ema' in kwargs.keys() and kwargs['ema'] == True:
            #     hyperplanes, mu= self.projection_pro.create_subspace([x_shot[::3][i], torch.cat((x_shot[1::3][i], x_shot[2::3][i]), dim=0)], shot_shape[1], shot_shape[2])
            # else:
            hyperplanes, mu= self.projection_pro.create_subspace([x_shot[::2][i], x_shot[1::2][i]], shot_shape[1], shot_shape[2])
            logits_i, dist_loss_i, _ = self.projection_pro.projection_metric([x_query[::2][i].double(), x_query[1::2][i]], hyperplanes)
            logits.append(logits_i)
            dist_loss.append(dist_loss_i)
            if kwargs['cts_loss_weight'] != 0:
                pos_set.append(torch.cat((x_shot[::2][i], x_query[::2][i]), dim=0))
                neg_set.append(torch.cat((x_shot[1::2][i], x_query[1::2][i]), dim=0))
            # elif kwargs['cts_loss_weight'] != 0 and 'ema' in kwargs and kwargs['ema'] == False:
            #     pos_set.append(x_shot[::2][i])
            #     neg_set.append(x_shot[1::2][i])
            # elif kwargs['cts_loss_weight'] != 0 and 'ema' in kwargs and kwargs['ema'] == True:
            #     pos_set.append(x_shot[::3][i])
            #     neg_set.append(x_shot[1::3][i])
            #     neg_set.append(x_shot[2::3][i])

        if kwargs['cts_loss_weight'] != 0:
            pos_set = torch.stack(pos_set, dim = 0)
            neg_set = torch.stack(neg_set, dim = 0).view(1, -1, x_shot.shape[-1])
            input_feat = torch.cat((pos_set, neg_set), dim=1)
            per_batch, _, _ = input_feat.shape
            # if 'ema' not in kwargs:
            cts_label = torch.cat((torch.zeros([per_batch, 7]), torch.ones([per_batch, 7])), dim=1)
            # elif kwargs['ema'] == False:
            #     cts_label = torch.cat((torch.zeros([per_batch, 6]), torch.ones([per_batch, 6])), dim=1)
            # else:
            #     cts_label = torch.cat((torch.zeros([per_batch, 6]), torch.ones([per_batch, 12])), dim=1)
            cts_loss, cts_loss_all = self.cts_loss_func(input_feat, cts_label)
        else:
            cts_loss = torch.as_tensor(0).cuda(non_blocking=True)
        
        
        

        logits = torch.stack(logits, dim = 0).view(-1, 2) * self.temp
        dist_loss = torch.sum(torch.stack(dist_loss, dim = 0))
        return logits, dist_loss, cts_loss




class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()

        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

