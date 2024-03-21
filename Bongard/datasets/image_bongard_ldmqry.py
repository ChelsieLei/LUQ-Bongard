# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Bongard-HOI. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------

import os
import json
from PIL import Image
import numpy as np
import glob
from PIL import ImageFilter
import random
import cv2
import pickle
import pdb

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# from detectron2
from detectron2.structures import Boxes
from detectron2.data import transforms as T

from .datasets import register


@register('image-bongard-ldmqry')
class ImageBongardAug(Dataset):

    def __init__(self, use_gt_bbox=False, image_size=256, box_size=256, **kwargs):
        self.bong_size = kwargs.get('bong_size')
        if self.bong_size is None:
            self.bong_size = 7
        if box_size is None:
            box_size = image_size

        split_file_path = kwargs.get('split_file')
        assert split_file_path is not None
        bongard_problems = json.load(open(split_file_path, 'r'))
        self.new_bongard = json.load(open('./cache/generated_train_query_selected_v4.json', 'r'))


        self.bongard_problems = bongard_problems
        self.im_dir = kwargs.get('im_dir')
        assert self.im_dir is not None
        self.n_tasks = len(bongard_problems)

        # bounding boxes info
        # use ground-truth boxes if not provided
        self.use_gt_bbox = use_gt_bbox
        if 'comb_gt_det' in kwargs.keys():
            self.comb_gt_det = kwargs.get('comb_gt_det')
        else:
            self.comb_gt_det = False
        bbox_file = kwargs.get('bbox_file')
        if not use_gt_bbox or self.comb_gt_det is True:
            with open(bbox_file, 'rb') as f:
                self.boxes_data = pickle.load(f)
            self.det_thresh = kwargs.get('det_thresh')
            if self.det_thresh is None:
                self.det_thresh = 0.7
            if 'use_DEKR' in kwargs and kwargs['use_DEKR'] is False:
                self.human_det_roi = None
            else:
                human_det_file = kwargs.get('human_det_file')
                with open(human_det_file, 'rb') as ff:
                    self.human_det_roi = pickle.load(ff)
                self.human_det_thres = kwargs.get('human_det_thres')
                self.iou_thres = kwargs.get('iou_thres')
        else:
            self.boxes_data = None

        self.do_aug = 'augment' in kwargs or 'augment_plus' in kwargs

        self.pix_mean = (0.485, 0.456, 0.406)
        self.pix_std = (0.229, 0.224, 0.225)

        # detectron2-style data augmentation
        sample_style = 'range'
        augmentations = [T.ResizeShortestEdge(image_size, int(image_size * 2), sample_style)]
        augmentation_rot = [T.ResizeShortestEdge(image_size, int(image_size * 2), sample_style)]
        if kwargs.get('augment') or kwargs.get('augment_plus'):
            augmentations.append(
                T.RandomFlip(
                    horizontal=True,
                    vertical=False,
                )
            )
            augmentation_rot.append(
                T.RandomFlip(
                    horizontal=True,
                    vertical=False,
                )
            )
            augmentation_rot.append(
                T.RandomRotation(
                    angle = [30, 90],
                    sample_style = "range"
                )
                
            )
            # T.RotationTransform()
        if kwargs.get('augment_plus'):
            self.photo_aug = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        else:
            self.photo_aug = None
        # self.augmentations = T.AugmentationList(augmentations)
        self.augmentations = augmentations
        self.augmentation_rot = augmentation_rot


    def get_crop_im(self, im_path, x1, y1, x2, y2, dilation=0.1, subject_dim=None, object_dim=None):
        im_path = os.path.join(self.im_dir, im_path)
        im = cv2.imread(im_path).astype(np.float32)
        # BGR to RGB
        im = im[:, :, ::-1]
        assert im is not None, im_path
        imh, imw, _ = im.shape

        if subject_dim is not None:
            sub_x1, sub_y1, sub_x2, sub_y2 = subject_dim
            im = cv2.rectangle(im, (sub_x1, sub_y1), (sub_x2, sub_y2), thickness=2, color=(0, 255, 0))
        if object_dim is not None:
            obj_x1, obj_y1, obj_x2, obj_y2 = object_dim
            im = cv2.rectangle(im, (obj_x1, obj_y1), (obj_x2, obj_y2), thickness=2, color=(0, 255, 0))

        h = y2 - y1
        w = x2 - x1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        # dilation
        h = (1 + dilation) * h
        w = (1 + dilation) * w
        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = min(imw, int(cx + w / 2))
        y2 = min(imh, int(cy + h / 2))
        crop_im = im[y1 : y2, x1 : x2]
        return crop_im, x1, y1, x2, y2

    def get_bbox(self, im, dim):
        imh, imw = im.height, im.width
        x1, y1, x2, y2 = dim
        if x1 <= 1 and y1 <= 1 and x2 <= 1 and y2 <= 1:
            x1 = imw * x1
            y1 = imh * y1
            x2 = imw * x2
            y2 = imh * y2
        return torch.Tensor(([x1, y1, x2, y2]))

    def get_detection_boxes_in_crop(self, image_id, x1, y1, x2, y2, obj_gt = None):
        all_boxes = self.boxes_data[image_id]['boxes']
        if obj_gt is not None:
            boxes = [obj_gt]
        else:
            boxes = []
        for idx in range(all_boxes.shape[0]):
            x1_i, y1_i, x2_i, y2_i = all_boxes[idx]
            if x1_i >= x1 and y1_i >= y1 and x2_i <= x2 and y2_i <= y2:
                boxes.append(torch.Tensor([x1_i - x1, y1_i - y1, x2_i - x1, y2_i - y1]))

        if len(boxes) != 0:
            boxes = torch.stack(boxes, dim=0)
        else:
            # use the entire image if no detections found
            boxes = torch.Tensor([0, 0, x2 - x1, y2 - y1])
        return boxes

    def iou(self, box1, box2):
        # box format: xyxy
        area1 = (box1[3] - box1[1]) * (box1[2] - box1[0])
        area2 = (box2[3] - box2[1]) * (box2[2] - box2[0])
        inter_area = (min(box1[2], box2[2]) - max(box1[0], box2[0])) * \
                    (min(box1[3], box2[3]) - max(box1[1], box2[1]))
        return inter_area / (area1 + area2 - inter_area)

    def get_human_det_in_crop(self, image_id, boxes, crop_size, subj_gt = None, obj_gt = None):
        # if image_id == 'HOI_PVP_00022857.jpg':
        #     print("wait")
        hei, wid, _ = crop_size
        ### get human roi
        all_location = self.human_det_roi[image_id]
        hum_num = len(all_location)
        
        boxes = boxes.tolist()
        boxes_cp = list(boxes)
        if subj_gt is not None:
            human_roi = [subj_gt.tolist()]
        else:
            human_roi = []
        for hum_id in range(hum_num):
            if len(all_location[hum_id][:]) <= 1:   ## only one point detection or empty detection
                continue
            score_list = [c[2] for c in all_location[hum_id][:]]
            proper_index = [i for i in range(len(score_list)) if score_list[i] >= self.human_det_thres]
            
            x_coord_list = [c[0] * wid for c in all_location[hum_id][:]]
            x_coord_list = [x_coord_list[i] for i in proper_index]
            y_coord_list = [c[1] * hei for c in all_location[hum_id][:]]
            y_coord_list = [y_coord_list[i] for i in proper_index]
            point_type_list = [c[3] for c in all_location[hum_id][:]]
            point_type_list = [point_type_list[i] for i in proper_index]
           
            ### boundary
            min_x = min(x_coord_list)
            max_x = max(x_coord_list)
            min_y = min(y_coord_list)
            max_y = max(y_coord_list)
            width_i = max_x - min_x
            height_i = max_y - min_y
            # ### boundary index to find related boundary key point type
            # min_x_pointtype = point_type_list[x_coord_list.index(min_x)]
            # max_x_pointtype = point_type_list[x_coord_list.index(max_x)]
            # min_y_pointtype = point_type_list[y_coord_list.index(min_y)]
            # max_y_pointtype = point_type_list[y_coord_list.index(max_y)]  
            
            if max(point_type_list) <=4 :  ### roughly whole body detected
                min_y = max(min_y - height_i * 2 , 0)
                max_y = min(max_y +  height_i *5, hei)
                min_x = max( min_x - width_i , 0)
                max_x = min(max_x + width_i , wid)
            else:
                min_y = max(min_y - height_i / 15, 0)
                max_y = min(max_y + height_i / 15, hei)
                min_x = max(min_x - width_i / 6, 0)
                max_x = min(max_x + width_i / 6, wid)                
            human_roi.append([min_x, min_y, max_x, max_y])
            ## exclude human from object boxes
            num_box_init = len(boxes)
            if len(np.array(boxes).shape) == 2 and np.array(boxes).shape[0] > 1:
                count = -1
                for i in range(num_box_init):
                    count += 1
                    iou_i = self.iou([min_x, min_y, max_x, max_y], boxes[count])
                    if iou_i > self.iou_thres:
                        boxes.pop(count)
                        count -= 1
                    if count == len(boxes) - 1:
                        break
        if len(boxes) == 0:
            if obj_gt is not None:
                boxes = obj_gt.tolist()
            else:
                boxes = boxes_cp
        if len(human_roi) == 0:
            if subj_gt is not None:
                human_roi = subj_gt.tolist()
            else:
                human_roi.append([0, 0, wid, hei])
        ## nms human_roi   human_list is sorted as the descent sequence of score
        num_human_init = len(human_roi)
        if (num_human_init) != 1:
            for hum_i in range(num_human_init):
                h_count = hum_i
                for hum_j in range(hum_i+1, num_human_init):
                    h_count += 1
                    iou_h_i = self.iou(human_roi[hum_i], human_roi[h_count])
                    if iou_h_i > self.iou_thres:
                        human_roi.pop(h_count)
                        h_count -= 1
                    if h_count == len(human_roi) - 1:
                        break
                if hum_i >= len(human_roi) - 2:
                    break
        
        if len(np.array(boxes).shape) != 2:
            boxes = torch.tensor(boxes).view(-1, 4)
        else:
            boxes = torch.tensor(boxes)
        all_boxes = torch.cat((torch.tensor(human_roi), boxes), dim=0)
        return all_boxes, len(human_roi)
    
    def get_triplet_crop(self, scene_graph, tp_idx, image_id, im_dir='../OpenImages/validation', show_bbox=False):
        tp = scene_graph['triplets'][tp_idx]
        objects = scene_graph['objects']
        sub_x1, sub_y1, sub_x2, sub_y2 = objects[tp['subject']]['dimension']
        obj_x1, obj_y1, obj_x2, obj_y2 = objects[tp['object']]['dimension']
        assert sub_x2 > 1 and sub_y2 > 1
        assert obj_x2 > 1 and obj_y2 > 1

        x1 = min(sub_x1, obj_x1)
        y1 = min(sub_y1, obj_y1)
        x2 = max(sub_x2, obj_x2)
        y2 = max(sub_y2, obj_y2)

        im_path = os.path.join(im_dir, image_id)
        if not im_path.endswith('.jpg') and not im_path.endswith('.png'):
            im_path += '.jpg'
        if show_bbox:
            crop_im, x1, y1, x2, y2 = self.get_crop_im(im_path, x1, y1, x2, y2, subject_dim=objects[tp['subject']]['dimension'], object_dim=objects[tp['object']]['dimension'])
        else:
            crop_im, x1, y1, x2, y2 = self.get_crop_im(im_path, x1, y1, x2, y2)

        if self.boxes_data is None:
            # use ground-truth bounding boxes
            sub_bbox = torch.Tensor([sub_x1 - x1, sub_y1 - y1, sub_x2 - x1, sub_y2 - y1])
            obj_bbox = torch.Tensor([obj_x1 - x1, obj_y1 - y1, obj_x2 - x1, obj_y2 - y1])
            boxes = torch.stack((sub_bbox, obj_bbox), dim=0)
        else:
            boxes = self.get_detection_boxes_in_crop(image_id, x1, y1, x2, y2)

        return crop_im, boxes

    def get_image(self, crop_info, rot_flag=False, new_q = False):
        im_path = crop_info['im_path']
        
        sub_bbox = crop_info['sub_bbox']
        obj_bbox = crop_info['obj_bbox']

        if new_q is True:
            # print("im_path", im_path)
            im_path = os.path.join("./cache/ldm_selected_v4", im_path)
        else:
            im_path = os.path.join(self.im_dir, im_path)
            x1, y1, x2, y2 = crop_info['crop_bbox']
        im = cv2.imread(im_path).astype(np.float32)
        imh, imw, _ = im.shape
        # BGR to RGB
        im = im[:, :, ::-1]
        assert im is not None, im_path
        
        if new_q is True:
            crop_im = im
        else:
        # fix image and annotation mismatch of openimages
            if "openimages" in im_path:
                # x1, y1, x2, y2 = int(1.6 * x1), int(1.6 * y1), int(1.6 * x2), int(1.6 * y2)
                # # if bounding box is out of the border, use the whole image
                # if y1 > imh or y2 > imh or x1 > imw or x2 > imw:
                #     x1, y1, x2, y2 = int(0), int(0), imw, imh
                x1 = min(int(1.6 * x1), imw)
                y1 = min(int(1.6 * y1), imh)
                x2 = min(int(1.6 * x2), imw)
                y2 = min(int(1.6 * y2), imh)
                if x1 == x2:
                    x1, x2 = int(0), imw
                if y1 == y2:
                    y1, y2 = int(0), imh
                sub_bbox[0] = min(sub_bbox[0] * 1.6, x2-x1)
                sub_bbox[1] = min(sub_bbox[1] * 1.6, y2-y1)
                sub_bbox[2] = min(sub_bbox[2] * 1.6, x2-x1)
                sub_bbox[3] = min(sub_bbox[3] * 1.6, y2-y1)
                obj_bbox[0] = min(obj_bbox[0] * 1.6, x2-x1)
                obj_bbox[1] = min(obj_bbox[1] * 1.6, y2-y1)
                obj_bbox[2] = min(obj_bbox[2] * 1.6, x2-x1)
                obj_bbox[3] = min(obj_bbox[3] * 1.6, y2-y1)

            crop_im = im[y1:y2, x1:x2]

        assert crop_im.shape[0]*crop_im.shape[1] != 0

        if self.boxes_data is None:
            # use ground-truth bounding boxes
            sub_bbox = torch.Tensor(sub_bbox)
            obj_bbox = torch.Tensor(obj_bbox)
            all_boxes = torch.stack((sub_bbox, obj_bbox), dim=0)
            hum_roi_num = 1
        elif self.comb_gt_det is True:
            image_id = os.path.basename(im_path)
            sub_bbox = torch.Tensor(sub_bbox)
            obj_bbox = torch.Tensor(obj_bbox)
            boxes = self.get_detection_boxes_in_crop(image_id, x1, y1, x2, y2, obj_gt = obj_bbox)
            if self.human_det_roi is not None:
                all_boxes, hum_roi_num = self.get_human_det_in_crop(image_id, boxes, crop_im.shape,subj_gt = sub_bbox)
            else:
                all_boxes = boxes
                hum_roi_num = 0
        else:

            ####provide gt bboxes for training
            if new_q is True:
                hum_roi_num = len(sub_bbox)
                if hum_roi_num == 0:
                    hum_roi_num = 1
                    sub_bbox = [[0, 0, imw, imh ]]
                if len(obj_bbox) == 0:
                    obj_bbox = [[0, 0, imw, imh ]]
            sub_bbox = torch.Tensor(sub_bbox).int()
            sub_bbox = torch.clamp(sub_bbox,min=0.0)
            obj_bbox = torch.Tensor(obj_bbox).int()
            obj_bbox = torch.clamp(obj_bbox,min=0.0)
            gt_boxes = torch.cat((sub_bbox, obj_bbox), dim=0)

            image_id = os.path.basename(im_path)
            if new_q is True:
                all_boxes = torch.cat((sub_bbox, obj_bbox), dim=0)
                
            else:
                boxes = self.get_detection_boxes_in_crop(image_id, x1, y1, x2, y2)
                if self.human_det_roi is not None:
                    all_boxes, hum_roi_num = self.get_human_det_in_crop(image_id, boxes, crop_im.shape)
                else:
                    all_boxes = boxes
                    hum_roi_num = 0            

        # return self.transform(img), boxes
        org_shape = crop_im.shape
        aug_input = T.AugInput(crop_im, boxes=all_boxes)
        if rot_flag == True:
            self.augmentation_list = T.AugmentationList(self.augmentation_rot)
        else:
            self.augmentation_list = T.AugmentationList(self.augmentations)
        transforms = self.augmentation_list(aug_input)
        crop_im, boxes = aug_input.image, (aug_input.boxes)
        aug_shape = crop_im.shape

        ##### transfer the gt_boxes
        # if new_q is False:
        #     y_ratio = aug_shape[0] / org_shape[0]
        #     x_ratio = aug_shape[1] / org_shape[1]
        #     gt_boxes = gt_boxes*(torch.as_tensor([[x_ratio, y_ratio, x_ratio, y_ratio]]).repeat(gt_boxes.shape[0],1))
        # else:
        #     gt_boxes = []


        if hum_roi_num != 0:
            human_roi = Boxes(boxes[:hum_roi_num])
            boxes = Boxes(boxes[hum_roi_num:])
        else:
            human_roi = None
            boxes = Boxes(boxes)
            
        if self.photo_aug is not None:
            # color jittering of the input image
            crop_im = np.array(self.photo_aug(Image.fromarray(crop_im.astype(np.uint8))), dtype=np.float32)
        for i in range(3):
            crop_im[:, :, i] = (crop_im[:, :, i] / 255. - self.pix_mean[i]) / self.pix_std[i]
        crop_im = torch.as_tensor(np.ascontiguousarray(crop_im.transpose(2, 0, 1)))
        

        boxes_tensor = boxes.tensor
        boxes_dim = [
            (boxes_tensor[:, 2] + boxes_tensor[:, 0]) / 2 / imw,  # cx
            (boxes_tensor[:, 3] + boxes_tensor[:, 1]) / 2 / imh,  # cy
            (boxes_tensor[:, 2] - boxes_tensor[:, 0]) / imw, # width
            (boxes_tensor[:, 3] - boxes_tensor[:, 1]) / imh, # height
        ]
        boxes_dim = torch.stack(boxes_dim, dim=1)

        rot_aug_cropim = torch.rot90(crop_im, 1, [1, 2])
        ### aug image boxes
        aug_boxes = Boxes(boxes.tensor.mm((torch.as_tensor([[0,0,0,-1], [1, 0,0,0], [0,-1,0,0],[0,0,1,0]]).float())) + \
            torch.as_tensor([0,crop_im.shape[2],0,crop_im.shape[2]]).repeat(boxes.tensor.shape[0],1))
        if human_roi is not None:
            aug_human_roi = Boxes(human_roi.tensor.mm((torch.as_tensor([[0,0,0,-1], [1, 0,0,0], [0,-1,0,0],[0,0,1,0]]).float())) + \
            torch.as_tensor([0,crop_im.shape[2],0,crop_im.shape[2]]).repeat(human_roi.tensor.shape[0],1))
        else:
            aug_human_roi = None
        return crop_im, boxes, boxes_dim, human_roi, gt_boxes, rot_aug_cropim, aug_boxes, aug_human_roi

    def pad_images(self, pos_ims, neg_ims, teach_neg_ims):
        max_imh, max_imw = -1, -1
        for im_i in pos_ims:
            _, imh, imw = im_i.shape
            max_imh = max(max_imh, imh)
            max_imw = max(max_imw, imw)

        for im_i in neg_ims:
            _, imh, imw = im_i.shape
            max_imh = max(max_imh, imh)
            max_imw = max(max_imw, imw)

        for im_i in teach_neg_ims:
            _, imh, imw = im_i.shape
            max_imh = max(max_imh, imh)
            max_imw = max(max_imw, imw)



        for idx, im_i in enumerate(pos_ims):
            pad_im_i = torch.zeros((3, max_imh, max_imw))
            _, imh, imw = im_i.shape
            pad_im_i[:, :imh, :imw] = im_i
            pos_ims[idx] = pad_im_i

        for idx, im_i in enumerate(neg_ims):
            pad_im_i = torch.zeros((3, max_imh, max_imw))
            _, imh, imw = im_i.shape
            pad_im_i[:, :imh, :imw] = im_i
            neg_ims[idx] = pad_im_i

        for idx, im_i in enumerate(teach_neg_ims):
            pad_im_i = torch.zeros((3, max_imh, max_imw))
            _, imh, imw = im_i.shape
            pad_im_i[:, :imh, :imw] = im_i
            teach_neg_ims[idx] = pad_im_i


        return pos_ims, neg_ims, teach_neg_ims
    
    def __len__(self):
        return len(self.bongard_problems)

    def __getitem__(self, i):
        """
        (Pdb) x_shot.shape                                                                                    │
        torch.Size([16, 2, 6, 3, 256, 256])                                                                   │
        (Pdb) x_query.shape                                                                                   │
        torch.Size([16, 2, 3, 256, 256])                                                                      │
        (Pdb) label_query.shape                                                                               │
        torch.Size([32])
        """
        pos_info_list, neg_info_list, task_name = self.bongard_problems[i]
        if str(i) in self.new_bongard.keys():
            new_q_list = self.new_bongard[str(i)]
        else:
            new_q_list = []


        pos_ims, pos_boxes, pos_boxes_dim, pos_hum_roi, pos_gt_boxes= [], [], [], [], []
        count = 0
        for pos_info_i in pos_info_list:
            count += 1
            im_i, boxes_i, _, hum_roi_i, gt_boxes_i, _, _, _ = self.get_image(pos_info_i)

            pos_ims.append(im_i)
            pos_boxes.append(boxes_i)
            # if hum_roi_i is not None:
            pos_hum_roi.append(hum_roi_i)
            # else:
            #     continue
            if count ==7 and len(new_q_list) == 0:
                im_i, boxes_i, _, hum_roi_i,  _, _, _, _ = self.get_image(pos_info_i, rot_flag = True)
                pos_ims.append(im_i)
                pos_boxes.append(boxes_i)
                pos_hum_roi.append(hum_roi_i)
        
        neg_ims, neg_boxes, neg_boxes_dim, neg_hum_roi, neg_gt_boxes = [], [], [], [], []
        neg_act = []

        zip_neg_act = {}
        count = 0
        for neg_info_i in neg_info_list:
            count += 1
            if neg_info_i['act_class'] not in zip_neg_act.keys():
                zip_neg_act[neg_info_i['act_class']] = []
            zip_neg_act[neg_info_i['act_class']].append(count)

        teach_neg_ims, teach_neg_boxes, teach_neg_hum_roi = [], [], []
        count = 0
        for neg_info_i in neg_info_list:
            count += 1

            im_i, boxes_i, _, hum_roi_i, _, _, _, _ = self.get_image(neg_info_i)
            if count != 7:
                teach_neg_ims.append(im_i)
                teach_neg_boxes.append(boxes_i)
                teach_neg_hum_roi.append(hum_roi_i)

            ### TODO correct negative selection here!!!!
            if (count == 7):
                pass
            elif len(zip_neg_act[neg_info_i['act_class']]) >1:
                _, _, _, _,  _, im_i, boxes_i, hum_roi_i = self.get_image(neg_info_i)

            neg_gt_boxes.append(gt_boxes_i)
            neg_act.append(neg_info_i['act_class'])
            neg_ims.append(im_i)
            neg_boxes.append(boxes_i)
            
            # if hum_roi_i is not None:
            neg_hum_roi.append(hum_roi_i)
            # else:
            #     continue
            if count ==7 and len(new_q_list) == 0:
                im_i, boxes_i, _, hum_roi_i,  _, _, _, _ = self.get_image(neg_info_i, rot_flag = True)
                neg_ims.append(im_i)
                neg_boxes.append(boxes_i)
                neg_hum_roi.append(hum_roi_i)
        

        if len(new_q_list) != 0:
            if len(new_q_list) == 2:
                temp_list = new_q_list
            else:
                temp = random.randint(0,len(new_q_list)-2)
                temp_list = [new_q_list[temp], new_q_list[temp+1]]
            for info_i in temp_list:
                im_i, boxes_i, _, hum_roi_i, _, _, _, _ = self.get_image(info_i, new_q = True)
                # print("new query number id ", int(info_i['im_path'].split("/")[-1][:-4]), info_i['im_path'])

                # if i == 1:
                #     print(torch.max(im_i), info_i)
                if int(info_i['im_path'].split("/")[-1][:-4])%2 == 0:
                    if torch.max(im_i) <= 0:
                        im_i, boxes_i, _, hum_roi_i,  _, _, _, _ = self.get_image(pos_info_list[-1], rot_flag = True)

                    pos_ims.append(im_i)
                    pos_boxes.append(boxes_i)
                    pos_hum_roi.append(hum_roi_i)
                else:
                    if torch.max(im_i) <= 0:
                        im_i, boxes_i, _, hum_roi_i,  _, _, _, _ = self.get_image(neg_info_list[-1], rot_flag = True)

                    neg_ims.append(im_i)
                    neg_boxes.append(boxes_i)
                    neg_hum_roi.append(hum_roi_i)                


        pos_ims, neg_ims, teach_neg_ims = self.pad_images(pos_ims, neg_ims, teach_neg_ims)

        pos_shot_ims = pos_ims[:6]
        pos_query_im = pos_ims[6]
        neg_shot_ims = neg_ims[:6]
        neg_query_im = neg_ims[6]
        pos_new_q_im = pos_ims[7:]
        neg_new_q_im = neg_ims[7:]


        if len(pos_hum_roi) != 0 and len(neg_hum_roi) != 0:
            pos_shot_hum_roi = pos_hum_roi[:6]
            pos_query_hum_roi =  pos_hum_roi[6]
            neg_shot_hum_roi = neg_hum_roi[:6]
            neg_query_hum_roi= neg_hum_roi[6]
            shot_human_roi = pos_shot_hum_roi + neg_shot_hum_roi
            query_human_roi = [pos_query_hum_roi, neg_query_hum_roi]

            teach_hum_roi = pos_shot_hum_roi.copy() + teach_neg_hum_roi
            
            pos_new_q_hum_roi = pos_hum_roi[7:]
            neg_new_q_hum_roi = neg_hum_roi[7:]
            if isinstance(pos_new_q_hum_roi, list):
                new_q_hum_roi = pos_new_q_hum_roi + neg_new_q_hum_roi
            else:
                new_q_hum_roi = [pos_new_q_hum_roi, neg_new_q_hum_roi]
            
        else:
            shot_human_roi = None
            query_human_roi = None
            shot_aug_humanroi = None

        pos_shot_boxes = pos_boxes[:6]
        pos_query_boxes =  pos_boxes[6]
        neg_shot_boxes = neg_boxes[:6]
        neg_query_boxes = neg_boxes[6]
        shot_boxes = pos_shot_boxes + neg_shot_boxes
        query_boxes = [pos_query_boxes, neg_query_boxes]

        teach_boxes = pos_shot_boxes.copy() + teach_neg_boxes

        pos_new_q_box = pos_boxes[7:]
        neg_new_q_box = neg_boxes[7:]
        if isinstance(pos_new_q_box, list):
            new_q_box = pos_new_q_box + neg_new_q_box
        else:
            new_q_box = [pos_new_q_box, neg_new_q_box]


        pos_shot_ims = torch.stack(pos_shot_ims, dim=0)
        neg_shot_ims = torch.stack(neg_shot_ims, dim=0)
        teach_neg_ims = torch.stack(teach_neg_ims, dim=0)
        shot_ims = torch.stack((pos_shot_ims, neg_shot_ims), dim=0)

        teach_ims = torch.stack((pos_shot_ims.clone(), teach_neg_ims), dim=0)

        query_ims = torch.stack((pos_query_im, neg_query_im), dim=0)
        query_labs = torch.Tensor([0, 1]).long()

        if len(pos_new_q_im) != 0 and len(neg_new_q_im) != 0:
            pos_new_q_im = torch.stack(pos_new_q_im, dim=0)
            neg_new_q_im = torch.stack(neg_new_q_im, dim=0)
            new_q_ims = torch.stack((pos_new_q_im, neg_new_q_im), dim=0)
        elif len(pos_new_q_im) == 0:
            new_q_ims = torch.stack(neg_new_q_im, dim=0)
            # print("no pos query", len(neg_new_q_im))
            # print("impath", pos_info_list[0])
        else:
            new_q_ims = torch.stack(pos_new_q_im, dim=0)
            # print("no neg query", len(pos_new_q_im))
            # print("impath", pos_info_list[0])
        # print("----------", new_q_ims.shape)
        if len(new_q_ims.shape) == 5:
            new_q_ims = new_q_ims.squeeze(1)
        
        torch.cuda.empty_cache()
        if len(set(neg_act)) == 1:   ## the number of HOI classes in the negative set
            flag = 1
        elif len(set(neg_act)) == 2:
            flag = 2
        elif len(set(neg_act)) == 3:
            flag = 3
        elif len(set(neg_act)) == 4:
            flag = 4
        elif len(set(neg_act)) == 5:
            flag = 5
        elif len(set(neg_act)) == 6:
            flag = 6
        elif len(set(neg_act)) == 7:
            flag = 7
        if neg_act[-1] in neg_act[:6]:  ### query in negative support or not
            q_flag = 0
        else:
            q_flag = 1

        ret_dict = {
            'shot_ims': shot_ims,
            'shot_boxes': shot_boxes,
            'query_ims': query_ims,
            'query_boxes': query_boxes,
            'query_labs': query_labs,
            'shot_boxes_dim': torch.Tensor([0]), #shot_boxes_dim,
            'query_boxes_dim': torch.Tensor([0]), #query_boxes_dim
            'shot_human_roi': shot_human_roi,
            'query_human_roi': query_human_roi,
            'task_name': task_name,
            "neg_act": neg_act,
            # "gt_boxes": gt_boxes,
            'teach_ims': teach_ims,
            'teach_boxes': teach_boxes,
            'teach_hum_roi': teach_hum_roi,
            'rot_query_ims': new_q_ims,
            'rot_query_box': new_q_box,
            'rot_query_humroi': new_q_hum_roi,
            'neg_query_act': neg_info_list[-1]['act_class'],
            'flag':flag,
            "q_flag": q_flag
        }

        return ret_dict


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def collate_images_boxes_dict(batch):
    def _pad_tensor(tensor_list):
        max_imh, max_imw = -1, -1
        for tensor_i in tensor_list:
            imh, imw = tensor_i.shape[-2], tensor_i.shape[-1]
            max_imh = max(max_imh, imh)
            max_imw = max(max_imw, imw)

        for idx, tensor_i in enumerate(tensor_list):
            pad_tensor_i = tensor_i.new_full(list(tensor_i.shape[:-2]) + [max_imh, max_imw], 0)
            imh, imw = tensor_i.shape[-2], tensor_i.shape[-1]
            pad_tensor_i[..., :imh, :imw].copy_(tensor_i)
            tensor_list[idx] = pad_tensor_i
        return tensor_list

    keys = list(batch[0].keys())
    batched_dict = {}
    for k in keys:
        data_list = []
        for batch_i in batch:
            data_list.append(batch_i[k])

        if isinstance(data_list[0], torch.Tensor):
            # print(k, len(data_list), data_list[0].shape)
            if len(data_list[0].shape) > 1:
                data_list = _pad_tensor(data_list)
            data_list = torch.stack(data_list, dim=0)
            # print("after", data_list.shape)
        batched_dict[k] = data_list
    return batched_dict

