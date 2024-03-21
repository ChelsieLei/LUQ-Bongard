# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Bongard-HOI. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------
'''
combine all the augmentation here
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models
import utils
from .models import register
from .subspace_projection import Subspace_Projection
from .SupCtsloss import SupConLoss
from .subspace_projection_transd import Subspace_transd_Projection



@register('my_model_transd')
class My_model(nn.Module):

    def __init__(self, encoder, encoder_args={}, method = 'cosine', type = 'raw'):
        super().__init__()
        print('encoder: {}'.format(encoder))
        print('encoder_args: {}'.format(encoder_args))
        self.encoder = models.make(encoder, **encoder_args)
        self.flag = 0
        if type == 'raw':  ##use f to build subspace
            self.projection_pro = Subspace_Projection(num_dim=5, type=method)  ### subspace dimension
        elif type == 'transd_space':
            self.projection_pro = Subspace_transd_Projection(num_dim=5, type=method)
            self.flag = 1
        if method == 'cosine':
            self.temp = nn.Parameter(torch.tensor(10.))
        elif method == 'l2_dist':
            self.temp = nn.Parameter(torch.tensor(0.1))
        self.cts_loss_func = SupConLoss()
        self.aug_param =(torch.tensor(0.4))
        # self.classifier = Classifier(1280, 2) ### for test time adaptation

    def forward(self, x_shot, x_query, **kwargs):  ## flag 0 normal subspace; flag 1 positive bias subspace
        if 'stage' in kwargs.keys():
            self.stage = kwargs['stage']
        else:
            self.stage = 1

        shot_shape = x_shot.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = self.encoder(x_shot, kwargs['shot_boxes'], kwargs['shot_human_roi'], kwargs['roi_position'], stage = 3)  ## only mix up ps with neg (no negative augmentatiaon)       
        x_shot = torch.stack(x_shot)     

        if 'query_boxes' in kwargs:
            assert 'shot_boxes' in kwargs
            x_query = self.encoder(x_query, kwargs['query_boxes'], kwargs['query_human_roi'],  kwargs['roi_position'])
        else:
            x_query = x_query.view(-1, *img_shape)
            x_query = self.encoder(x_query)
        if isinstance(x_query,list):
            x_query = torch.stack(x_query)   
        else:
            x_query = x_query.unsqueeze(0)



        logits, dist_loss, cts_loss = [], [], []
        mix_consistent_loss = []
        for k in range(shot_shape[0]):
            hyperplanes, mu= self.projection_pro.create_subspace([x_shot[::3][k], x_shot[1::3][k]], shot_shape[1], shot_shape[2])
            logits_i, dist_loss_i,_ = self.projection_pro.projection_metric([x_query[::2][k].double(), x_query[1::2][k]], hyperplanes,mu)
            mix_logits_i, _, mix_dist = self.projection_pro.projection_metric([x_shot[2::3][k].double()], hyperplanes,mu)
            logits.append(logits_i)
            dist_loss.append(dist_loss_i)

            q1, q2 = torch.argmax(logits_i, dim=1)[0].item(), torch.argmax(logits_i, dim=1)[1].item()
            if q1 == 0 and q2 == 0:
                hyperplanes, mu = self.projection_pro.create_subspace([torch.cat((x_shot[3*k], x_query[2*k:2*k+2].squeeze(1)), dim = 0), 
                                                                    x_shot[3*k+1]], 
                                                                    shot_shape[1], shot_shape[2])
                logits_mix, _, mix_dist_new = self.projection_pro.projection_metric([x_shot[3*k+2]], hyperplanes,mu)
            elif q1 == 0 and q2 == 1:

                # logits_1, _ = self.projection_pro.projection_metric([x_query[2*k], x_query[2*k+1]], hyperplanes,mu)
                hyperplanes, mu = self.projection_pro.create_subspace([torch.cat((x_shot[3*k], x_query[2*k]), dim = 0), 
                                                                    torch.cat((x_shot[3*k+1], x_query[2*k+1]), dim = 0)],
                                                                    shot_shape[1], shot_shape[2])
                logits_mix, _, mix_dist_new = self.projection_pro.projection_metric([x_shot[3*k+2]], hyperplanes,mu)
            elif q1 == 1 and q2 == 0:
                hyperplanes, mu = self.projection_pro.create_subspace([torch.cat((x_shot[3*k], x_query[2*k+1]), dim = 0), 
                                                                    torch.cat((x_shot[3*k+1], x_query[2*k]), dim = 0)],
                                                                    shot_shape[1], shot_shape[2])
                logits_mix, _, mix_dist_new = self.projection_pro.projection_metric([x_shot[3*k+2]], hyperplanes,mu)
            else:
                hyperplanes, mu = self.projection_pro.create_subspace([x_shot[3*k], 
                                                                    torch.cat((x_shot[3*k+1], x_query[2*k:2*k+2].squeeze(1)), dim = 0)],
                                                                    shot_shape[1], shot_shape[2])
                logits_mix, _, mix_dist_new = self.projection_pro.projection_metric([x_shot[3*k+2]], hyperplanes,mu)
            mix_consistent_loss_i = torch.norm((logits_mix - mix_logits_i))
            mix_consistent_loss.append(mix_consistent_loss_i)
            # logits_2, _ = self.projection_pro.projection_metric([x_query[2*k], x_query[2*k+1]], hyperplanes,mu)    


            # logits_1 = (logits_1[0] / (logits_1[0] + logits_1[1])) * torch.sign(logits_1[0])
            # logits_2 = (logits_2[0] / (logits_2[0] + logits_2[1])) * torch.sign(logits_2[0])
            # if logits_1 > logits_2:
            #     logits_i = torch.as_tensor([[1,0],[0,1]])
            # else:
            #     logits_i = torch.as_tensor([[0,1],[1,0]])
            # logits_new.append(logits_i)

        # mix_logits = []
        pos_set, neg_set = [], []
        for i in range(shot_shape[0]):   ### process each instance in the batch seperately
            if kwargs['cts_loss_weight'] != 0:
                pos_set.append(torch.cat((x_shot[::3][i], x_query[::2][i]), dim=0))
                neg_set.append(torch.cat((x_shot[1::3][i], x_query[1::2][i]), dim=0))
                neg_len = x_shot[1::3][i].shape[0]

            # hyperplanes, mu= self.projection_pro.create_subspace([x_shot[::3][i], x_shot[1::3][i]], shot_shape[1], shot_shape[2])
            # logits_i, dist_loss_i = self.projection_pro.projection_metric([x_query[::2][i].double(), x_query[1::2][i]], hyperplanes,mu)
            # mix_logits_i, _ = self.projection_pro.projection_metric([x_shot[2::3][i].double()], hyperplanes,mu)
            # mix_logits.append(mix_logits_i)
            # logits.append(logits_i)
            # dist_loss.append(dist_loss_i)

        if kwargs['cts_loss_weight'] != 0:
            pos_set = torch.stack(pos_set, dim = 0)
            neg_set = torch.stack(neg_set, dim = 0)
            input_feat = torch.cat((pos_set, neg_set), dim=1)
            per_batch, _, _ = input_feat.shape
            cts_label = torch.cat((torch.zeros([per_batch, 7]), torch.ones([per_batch, neg_len+1])), dim=1)
            cts_loss, cts_loss_all = self.cts_loss_func(input_feat, cts_label)
        else:
            cts_loss = torch.as_tensor(0).cuda(non_blocking=True) 

        logits = torch.stack(logits, dim = 0).view(-1, 2) * self.temp
        # mix_logits = torch.stack(mix_logits, dim = 0).view(-1, 2) * self.temp
        dist_loss = torch.sum(torch.stack(dist_loss, dim = 0))
        mix_consistent_loss = torch.sum(torch.stack(mix_consistent_loss, dim = 0))
        torch.cuda.empty_cache() 
        return logits, dist_loss, cts_loss, mix_consistent_loss



