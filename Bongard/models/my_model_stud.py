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
import pdb
import models
import utils
from .models import register
from .subspace_projection import Subspace_Projection
from .SupCtsloss import SupConLoss
import scipy.stats
from copy import deepcopy


@register('my_model_stud')
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

        # print(x_query.shape, kwargs['rot_query_ims'].shape)

        assert 'query_boxes' in kwargs
        # if 'ema' in kwargs.keys() and kwargs['ema'] == True:
        #     # print(kwargs['teach_neg_ims'].shape, x_shot.shape)
        #     x_shot =  torch.stack((x_shot[:,0], kwargs['teach_neg_ims']), dim=1)
        #     # x_shot =  torch.cat((x_shot[:,0].unsqueeze(1), kwargs['teach_neg_ims'].view(2, -1, *img_shape).unsqueeze(0)), dim=1)
        #     # print("before encode x_shot: ",x_shot.shape)
        #     x_shot = self.encoder(x_shot, [kwargs['shot_boxes'][i][:6] + kwargs['teach_neg_boxes'][i] for i in range(shot_shape[0])], 
        #                           [kwargs['shot_human_roi'][i][:6] + kwargs['teach_neg_hum_roi'][i] for i in range(shot_shape[0])], kwargs['roi_position'])
        # else:    
        x_shot = self.encoder(x_shot, kwargs['shot_boxes'], kwargs['shot_human_roi'], kwargs['roi_position'])  ##[pos_i, neg_i, pos_(i+1), neg_(i+1), ...]

        if isinstance(x_shot, list):
            x_shot = torch.stack(x_shot)     
        if 'query_boxes' in kwargs:
            assert 'shot_boxes' in kwargs
            # print("x_query" , x_query.shape)
            x_query = self.encoder(x_query, kwargs['query_boxes'], kwargs['query_human_roi'],  kwargs['roi_position'])
        if isinstance(x_query, list):
            x_query = torch.stack(x_query)   

        # if 'aug_query' in kwargs.keys():
        #     aug_query = self.encoder(kwargs['aug_query'], deepcopy(kwargs['query_boxes']), deepcopy(kwargs['query_human_roi']),  kwargs['roi_position'])
        #     aug_query = torch.stack(aug_query)
        
        if ('eval' not in kwargs.keys()) and 'rot_query_ims' in kwargs.keys():  
            rot_query = self.encoder(kwargs['rot_query_ims'], kwargs['rot_query_box'], kwargs['rot_query_humroi'],  kwargs['roi_position'])
            if isinstance(rot_query, list):
                rot_query = torch.stack(rot_query)

        logits, dist_loss, cts_loss, aug_logits = [], [], [], []
        pos_set, neg_set = [], []
        rot_q_logits = []
        for i in range(shot_shape[0]):   ### process each instance in the batch seperately
            hyperplanes, mu= self.projection_pro.create_subspace([x_shot[::2][i], x_shot[1::2][i]], shot_shape[1], shot_shape[2])

            if 'eval' not in kwargs.keys() and 'rot_query_ims' in kwargs.keys():  
                # if 'aug_query' in kwargs.keys():

                #     logits_i, dist_loss_i ,rot_logits_i= self.projection_pro.projection_metric([torch.cat((x_query, aug_query), dim=1)[::2][i].double(),
                #                                                                 torch.cat((x_query, aug_query), dim=1)[1::2][i]], hyperplanes,
                #                                                                 [rot_query[::2][i].double(), rot_query[1::2][i]])
                #     aug_logits.append(logits_i[:,:,1])
                #     logits.append(logits_i[:,:,0])
                # else:
                logits_i, dist_loss_i ,rot_logits_i = self.projection_pro.projection_metric([x_query[::2][i].double(), x_query[1::2][i]], hyperplanes,
                                                                                        [rot_query[::2][i].double(), rot_query[1::2][i]])
                logits.append(logits_i)            
                rot_q_logits.append(rot_logits_i) 
            else:
                logits_i, dist_loss_i,_ = self.projection_pro.projection_metric([x_query[::2][i].double(), x_query[1::2][i]], hyperplanes)
                logits.append(logits_i)
            dist_loss.append(dist_loss_i)


            if kwargs['cts_loss_weight'] != 0:
                pos_set.append(torch.cat((x_shot[::2][i], x_query[::2][i]), dim=0))
                neg_set.append(torch.cat((x_shot[1::2][i], x_query[1::2][i]), dim=0))
            # else:
            #     pos_set.append(x_shot[::2][i])
            #     neg_set.append(x_shot[1::2][i])

        if kwargs['cts_loss_weight'] != 0:
            pos_set = torch.stack(pos_set, dim = 0).view(1, -1, x_shot.shape[-1])
            neg_set = torch.stack(neg_set, dim = 0).view(1, -1, x_shot.shape[-1])
            input_feat = torch.cat((pos_set, neg_set), dim=1)
            per_batch, num_sample, _ = input_feat.shape

            # if 'ema' in kwargs:
            #     cts_label = torch.cat((torch.zeros([per_batch, 6]), torch.ones([per_batch, 6])), dim=1)
            # else:
            cts_label = torch.cat((torch.zeros([per_batch, int(num_sample/2)]), torch.ones([per_batch, int(num_sample/2)])), dim=1)
            cts_loss, cts_loss_all = self.cts_loss_func(input_feat, cts_label)
        else:
            cts_loss = torch.as_tensor(0).cuda(non_blocking=True)
        
        # print("----------", torch.stack(logits, dim = 0).shape)
        logits = torch.stack(logits, dim = 0).view(-1, 2) * self.temp
        dist_loss = torch.sum(torch.stack(dist_loss, dim = 0))
        if 'eval' not in kwargs.keys() and 'rot_query_ims' in kwargs.keys():  
            rot_q_logits = torch.stack(rot_q_logits, dim = 0).view(-1, 2) * self.temp
            # if 'aug_query' in kwargs.keys():
            #     aug_logits = torch.stack(aug_logits, dim = 0).view(-1, 2) * self.temp
            #     return logits, dist_loss, cts_loss, rot_q_logits, aug_logits

        return logits, dist_loss, cts_loss, rot_q_logits

        
