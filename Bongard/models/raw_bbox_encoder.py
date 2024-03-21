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
from random import shuffle
@register('raw_bbox_encoder')
class RelationnetBBoxNetworkEncoder(nn.Module):
    def __init__(self, encoder, **kwargs):
        super(RelationnetBBoxNetworkEncoder, self).__init__()

        # image encoder
        encoder = models.make(encoder)
        self.encoder = encoder
        self.proj = nn.Conv2d(encoder.out_dim, encoder.out_dim // 2, kernel_size=1)

        # ROI Pooler
        self.roi_pooler = ROIPooler(
           output_size=7,
           scales=(1/32,), # TODO: this works for resnet50
           sampling_ratio=0,
           pooler_type='ROIAlignV2',
        )
        self.roi_processor = nn.Sequential(
            nn.Conv2d(encoder.out_dim // 2, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*7*7, 1024),
            nn.ReLU(),
            # nn.Linear(1024, 512),
            # nn.ReLU()
        )
        self.roi_processor_ln = nn.LayerNorm(1024)
        rn_in_planes = 1024 * 2

        # bbox coord encoding
        self.roi_processor_box = nn.Linear(4, 256)
        self.roi_processor_box_ln = nn.LayerNorm(256)
        rn_in_planes = (1024 + 256) * 2        

        # relational encoding
        self.g_mlp = nn.Sequential(
            nn.Linear(rn_in_planes, rn_in_planes // 2),
            nn.ReLU(),
            nn.Linear(rn_in_planes // 2, rn_in_planes // 2),
            nn.ReLU(),
            nn.Linear(rn_in_planes // 2, rn_in_planes // 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.out_dim = rn_in_planes // 2
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.aug_fuse = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(2, 1, 1, bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d(1, 2, 1, bias=False),
        #     nn.Sigmoid())
        

    def process_single_image_rois(self, roi_feats):
        # relational encoding
        M, C = roi_feats.shape
        b = 1
        # 1xMxC
        x_flat = roi_feats.unsqueeze(0)

        # adapted from https://github.com/kimhc6028/relational-networks/blob/master/model.py
        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)  # (b * 1 * M * c)
        x_i = x_i.repeat(1, M, 1, 1)  # (b * M * M  * c)
        x_j = torch.unsqueeze(x_flat, 2)  # (b * M * 1 * c)
        x_j = x_j.repeat(1, 1, M, 1)  # (b * M * M  * c)

        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3) # (b * M * M  * 2c)
        # reshape for passing through network
        x_full = x_full.view(b * M * M, -1)  # (b*M*M)*2c
        x_g = self.g_mlp(x_full)
        # reshape again and sum
        x_g = x_g.view(b, M * M, -1)
        x_g = x_g.sum(1)
        return x_g

    def process_single_image_rois_hum(self,hum_roi_feats,  obj_roi_feats):
        # relational encoding
        M, _ = hum_roi_feats.shape
        N,_ = obj_roi_feats.shape
        b = 1
        # 1xMxC
        hum_roi_feats = hum_roi_feats.unsqueeze(0)
        obj_roi_feats = obj_roi_feats.unsqueeze(0)
        # adapted from https://github.com/kimhc6028/relational-networks/blob/master/model.py
        # cast all pairs against each other
        x_i = torch.unsqueeze(hum_roi_feats, 1)  # (b * 1 * M * c)
        x_i = x_i.repeat(1, N, 1, 1)  # (b * M * M  * c)
        x_j = torch.unsqueeze(obj_roi_feats, 2)  # (b * M * 1 * c)
        x_j = x_j.repeat(1, 1, M, 1)  # (b * M * M  * c)

        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3) # (b * M * M  * 2c)
        # reshape for passing through network
        x_full = x_full.view(b * M * N, -1)  # (b*M*M)*2c
        x_g = self.g_mlp(x_full)
        # reshape again and sum
        x_g = x_g.view(b, M * N, -1)
        x_g = x_g.sum(1)
        return x_g

    def create_mask(self, img, obj_boxes, hum_boxes, ratio  = 3):
        # _, imh, imw = img.shape
        for i in range(obj_boxes.shape[0]):
            x1, y1, x2, y2 = obj_boxes[i]
            if (y2-y1) < ratio or (x2-x1) < ratio:
                continue
            mask_height = int((y2 - y1)/ratio)
            mask_width = int((x2 - x1)/ratio)
            mask_x = torch.randint(int(x1), int(x2 - mask_width), (1, 1)).item()
            mask_y = torch.randint(int(y1), int(y2 - mask_height), (1, 1)).item()
            img[:, mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 255
            break

        for i in range(hum_boxes.shape[0]):
            x1, y1, x2, y2 = hum_boxes[i]
            if (y2-y1) < ratio or (x2-x1) < ratio:
                continue
            mask_height = int((y2 - y1)/ratio)
            mask_width = int((x2 - x1)/ratio)
            mask_x = torch.randint(int(x1), int(x2 - mask_width), (1, 1)).item()
            mask_y = torch.randint(int(y1), int(y2 - mask_height), (1, 1)).item()
            img[:, mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 255
            break
        return img


    def forward(self, im, boxes, human_roi, roi_position=False, stage = 1, augind = None, neg_act = None, gt_bbox = None, rot_neg_ims = None,\
                            rot_neg_boxes = None, rot_neg_hum_roi = None):
        # assert im.shape[0] == len(boxes), 'im: {} vs boxes: {}'.format(im.shape[0], len(boxes))
        img_shape = im.shape
    # if stage == 1:
        if len(img_shape) < 6:
            img_shape = im.unsqueeze(dim = 2).shape

        if stage == 3:  ###### mixup
            mix_im = torch.zeros((img_shape[0], img_shape[2], img_shape[-3], img_shape[-2], img_shape[-1])).to(im.device)
            for i in range(img_shape[0]):
                # if i %12 < 6: #### positive images
                #     continue
                mix_im[i] = im[i, 0] * 0.5 + im[i, 1] * 0.5  
                # obj_bb_i = torch.cat(boxes[int(i/12)][i%12].tensor, boxes[int(i/12)][i%12-6].tensor, dim = 1)
                # hum_bb_i = torch.cat(human_roi[int(i/12)][i%12].tensor, human_roi[int(i/12)][i%12-6].tensor, dim = 1)
                for k in range(6):
                    boxes[i].append(Boxes(torch.cat((boxes[i][k].tensor, boxes[i][k+6].tensor))))
                    human_roi[i].append(Boxes(torch.cat((human_roi[i][k].tensor, human_roi[i][k+6].tensor))))
            mix_im = mix_im.unsqueeze(1)
            im = torch.cat((im, mix_im), dim = 1)    
            img_shape = im.shape        

        if stage == 6:  ##### add mixup negative (different actions)
            im = im.view(img_shape[0], -1, *img_shape[-3:])
            final_im = []
            for i in range(img_shape[0]):
                final_im.append(im[i])
            new_objb = []
            new_humb = []
            augim = []
            aug_count = [] ### store negative augmentation number in each instance
            for i in range(img_shape[0]):
                count = 0
                act_list = {}
                for j in range(6, 12):
                    if neg_act[i][j-6] not in act_list.keys():
                        act_list[neg_act[i][j-6]] = j
                    else:
                        for k in (list(act_list.keys())):
                            if neg_act[i][j-6] != k:
                                im_i = final_im[i][j] *0.5 + final_im[i][act_list[k]] * 0.5      
                                boxes[i].append(Boxes(torch.cat((boxes[i][j].tensor, boxes[i][act_list[k]].tensor))))
                                human_roi[i].append(Boxes(torch.cat((human_roi[i][j].tensor, human_roi[i][act_list[k]].tensor))))
                                augim.append(im_i)
                                final_im[i] = torch.cat((final_im[i], im_i.unsqueeze(0)), dim = 0)
                                count += 1
                aug_count.append(count)
            im = torch.stack(final_im)
            del final_im


        im = im.view(-1, *img_shape[-3:])

        if stage == 2:  ##### occlusion
            new_objb = []
            new_humb = []
            if augind is not None:
                for i in range(img_shape[0]):
                    im_i = im[2*i]
                    obj_bb_i = boxes[i][augind].tensor
                    hum_bb_i = human_roi[i][augind].tensor
                    im[2*i] = self.create_mask(im_i, obj_bb_i, hum_bb_i)
                    
                    im_i = im[2*i + 1]
                    obj_bb_i = boxes[i][augind + 6].tensor
                    hum_bb_i = human_roi[i][augind + 6].tensor
                    im[2*i+1] = self.create_mask(im_i, obj_bb_i, hum_bb_i)
                    obj_i = [boxes[i][augind], boxes[i][augind + 6]]
                    hum_i = [human_roi[i][augind], human_roi[i][augind + 6]]
                    new_objb.append(obj_i)
                    new_humb.append(hum_i)
                boxes = new_objb
                human_roi = new_humb
                del new_objb, new_humb
            else:
                for i in range(im.shape[0]):
                    if i %12 < 6: #### positive images
                        continue
                    if neg_act is not None:
                        if i%6 == 0:
                            act_list = []
                        if neg_act[int(i/12)][int(i%6)] not in act_list:
                            act_list.append(neg_act[int(i/12)][int(i%6)])
                        else:
                            im_i = im[i]       
                            obj_bb_i = boxes[int(i/12)][i%12].tensor
                            hum_bb_i = human_roi[int(i/12)][i%12].tensor
                            im[i] = self.create_mask(im_i, obj_bb_i, hum_bb_i)
                                          
        if roi_position:
            enc_x = self.encoder(im, roi_position)
            x = self.avgpool(enc_x)
            x = torch.flatten(x, 1)
        else:
            # print("0")
            x = self.encoder(im, roi_position)
        if len(x.shape) != 4:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.proj(x)

        ############## object RoI pooling/align
        obj_boxes = []
        for boxes_i in boxes:
            # if stage == 1:
            obj_boxes.extend(boxes_i)
            # else:
            #     obj_boxes.append(boxes_i)
        num_obj_boxes = [boxes_i.tensor.shape[0] for boxes_i in obj_boxes]  ### each img bboxes numbers list
        if roi_position:
            roi_obj_feats = self.roi_pooler([self.proj(enc_x)], obj_boxes)
        else:
            roi_obj_feats = self.roi_pooler([x], obj_boxes)
        roi_obj_feats = self.roi_processor(roi_obj_feats)
        roi_obj_feats = self.roi_processor_ln(roi_obj_feats)
        
        obj_bbox_tensor = torch.cat([box.tensor for box in obj_boxes]).to(roi_obj_feats)
        # bbox coord normalization
        obj_bbox_tensor[:, 0] = obj_bbox_tensor[:, 0] / im.shape[3]
        obj_bbox_tensor[:, 1] = obj_bbox_tensor[:, 1] / im.shape[2]
        obj_bbox_tensor[:, 2] = obj_bbox_tensor[:, 2] / im.shape[3]
        obj_bbox_tensor[:, 3] = obj_bbox_tensor[:, 3] / im.shape[2]
        obj_bbox_tensor = obj_bbox_tensor * 2 - 1 ## x1,y1,x2,y2 change to the distance relative to image center
        obj_roi_box_feats = self.roi_processor_box_ln(self.roi_processor_box(obj_bbox_tensor))      
        roi_obj_feats = torch.cat([roi_obj_feats, obj_roi_box_feats], dim=-1)  
        
        ############## human RoI pooling/align
        if (isinstance(human_roi[0], list) and human_roi[0][0] is not None):
            hum_boxes = []
            for boxes_i in human_roi:
                # if stage == 1:
                hum_boxes.extend(boxes_i)
                # else:
                #     hum_boxes.append(boxes_i)
            num_hum_boxes = [boxes_i.tensor.shape[0] for boxes_i in hum_boxes]
            if roi_position:
                roi_hum_feats = self.roi_pooler([self.proj(enc_x)], hum_boxes)
            else:
                roi_hum_feats = self.roi_pooler([x], hum_boxes)
            roi_hum_feats = self.roi_processor(roi_hum_feats)
            roi_hum_feats = self.roi_processor_ln(roi_hum_feats)

            human_bbox_tensor = torch.cat([box.tensor for box in hum_boxes]).to(roi_hum_feats)
            
            # bbox coord normalization
            human_bbox_tensor[:, 0] = human_bbox_tensor[:, 0] / im.shape[3]
            human_bbox_tensor[:, 1] = human_bbox_tensor[:, 1] / im.shape[2]
            human_bbox_tensor[:, 2] = human_bbox_tensor[:, 2] / im.shape[3]
            human_bbox_tensor[:, 3] = human_bbox_tensor[:, 3] / im.shape[2]
            human_bbox_tensor = human_bbox_tensor * 2 - 1 ## x1,y1,x2,y2 change to the distance relative to image center
            hum_roi_box_feats = self.roi_processor_box_ln(self.roi_processor_box(human_bbox_tensor))      
            roi_hum_feats = torch.cat([roi_hum_feats, hum_roi_box_feats], dim=-1)

            feats_list = []
            start_obj_idx = 0
            start_hum_idx = 0
            count = -1
            for num_obj_boxes_i, num_hum_roi_i in zip(num_obj_boxes, num_hum_boxes):
                count += 1
                # num_hum_roi_i = num_hum_boxes[count]
                end_obj_idx = start_obj_idx + num_obj_boxes_i
                end_hum_idx = start_hum_idx + num_hum_roi_i
                try:
                    feats_list.append(self.process_single_image_rois_hum(roi_hum_feats[start_hum_idx:end_hum_idx]
                                                                    , roi_obj_feats[start_obj_idx:end_obj_idx]))
                except Exception as e:
                    print("error inside start_hum_idx, end_hum_idx", start_hum_idx, end_hum_idx, roi_hum_feats[start_hum_idx:end_hum_idx].shape)
                    print("start_obj_idx:end_obj_idx", start_obj_idx, end_obj_idx, roi_obj_feats[start_obj_idx:end_obj_idx].shape)
                start_obj_idx = end_obj_idx
                start_hum_idx = end_hum_idx
        else:
            roi_hum_feats = None
            feats_list = []
            start_obj_idx = 0
            count = -1
            for num_obj_boxes_i in num_obj_boxes:
                count += 1
                # num_hum_roi_i = num_hum_boxes[count]
                end_obj_idx = start_obj_idx + num_obj_boxes_i
                feats_list.append(self.process_single_image_rois(roi_obj_feats[start_obj_idx:end_obj_idx]))
                start_obj_idx = end_obj_idx
        # if stage == 2:
        #     return feats_list
        
        feats = []
        for i in range(img_shape[0]*img_shape[1]):
            feats.append(torch.cat(feats_list[i*img_shape[2]: (i+1)*img_shape[2]], dim=0))
        return feats


if __name__ == '__main__':
    im = torch.rand((8, 3, 128, 128))

    model = RelationnetBBoxNetworkEncoder(encoder='resnet50')
    x = model(im)
    print(x.shape)
