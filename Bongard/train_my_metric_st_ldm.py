# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#

# This work is licensed under the NVIDIA Source Code License
# for Bongard-HOI. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------

import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from pickle import FALSE
from torch.utils import data
import yaml
import numpy as np
import random
import time
from copy import deepcopy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from tqdm import tqdm
import math

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.image_bongard_ldmqry import collate_images_boxes_dict

from detectron2.structures import Boxes
from torch.autograd import Variable
import pdb
global_step = 0

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:32'
def main(config):
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args.gpu = ''#[i for i in range(torch.cuda.device_count())]
    if args.split_gpu == False:
        args.train_gpu = [i for i in range(torch.cuda.device_count())]
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus - 1):
            args.gpu += '{},'.format(i)
        args.gpu += '{}'.format(num_gpus - 1)
    elif ',' in args.train_gpu:
        args.train_gpu = [int(i) for i in (args.train_gpu.split(','))]
        count = 0
        for i in args.train_gpu:
            count +=1
            if count == len(args.train_gpu):
                args.gpu += '{}'.format(i)
            else:
                args.gpu += '{},'.format(i)
    else:
        args.train_gpu = [int(args.train_gpu)]
        args.gpu = '{}'.format(int(args.train_gpu[0]))

    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu
    utils.set_gpu(args.gpu)
    args.config = config

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.sync_bn = True
        port = utils.find_free_port()
        args.dist_url = args.dist_url.format(port)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    config = args.config
    svname = args.name
    if svname is None:
        config_name, _ = os.path.splitext(os.path.basename(args.config_file))
        svname = '{}shot'.format(config['n_shot'])
        svname += '_' + config['stud_model']
        if config['stud_model_args'].get('encoder'):
            svname += '-' + config['stud_model_args']['encoder']
        svname = os.path.join(config_name, config['train_dataset'], svname)
    if not args.test_only:
        svname += '-seed' + str(args.seed)
    if args.tag is not None:
        svname += '_' + args.tag

    sub_dir_name = 'default'
    if args.opts:
        sub_dir_name = args.opts[0]
        split = '#'
        for opt in args.opts[1:]:
            sub_dir_name += split + opt
            split = '#' if split == '_' else '_'
    svname = os.path.join(svname, sub_dir_name)

    if utils.is_main_process() and not args.test_only:
        save_path = os.path.join(args.save_dir, svname)
        utils.ensure_path(save_path, remove=False)
        utils.set_log_path(save_path)
        writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
        # writer.close()
        args.writer = writer
        

        yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

        logger = utils.Logger(file_name=os.path.join(save_path, "log_sdout.txt"), file_mode="a+", should_flush=True)
    else:
        save_path = os.path.join(args.save_dir, svname)
        utils.ensure_path(save_path, remove=False)
        utils.set_log_path(save_path)
        # save_path = None
        writer = None
        args.writer = writer
        logger = utils.Logger(file_name=os.path.join(save_path, "log_test.txt"), file_mode="a+", should_flush=True)

    #### Dataset ####

    n_way, n_shot = config['n_way'], config['n_shot']
    n_query = config['n_query']

    if config.get('n_train_way') is not None:
        n_train_way = config['n_train_way']
    else:
        n_train_way = n_way
    if config.get('n_train_shot') is not None:
        n_train_shot = config['n_train_shot']
    else:
        n_train_shot = n_shot
    if config.get('ep_per_batch') is not None:
        ep_per_batch = config['ep_per_batch']
    else:
        ep_per_batch = 1

    args.n_train_way = n_train_way
    args.n_train_shot = n_train_shot
    args.n_query = n_query
    args.n_shot = n_shot
    args.n_way = n_way
    args.stage = int(args.stage)

    # train
    dataset_configs = config['train_dataset_args']
    dataset_configs['use_gt_bbox'] = config['use_gt_bbox']
    if 'comb_gt_det' in config.keys():
        dataset_configs['comb_gt_det'] = config['comb_gt_det']
    train_dataset = datasets.make(config['train_dataset'], **dataset_configs)
    if utils.is_main_process():
        utils.log('train dataset: {} samples'.format(len(train_dataset)))
    if args.distributed:
        args.batch_size = int(ep_per_batch / ngpus_per_node)
        args.batch_size_val = int(ep_per_batch / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    else:
        args.batch_size = ep_per_batch
        args.batch_size_val = ep_per_batch
        args.workers = args.workers

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=False,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_images_boxes_dict
    )

    # val & test
    if args.test_only:
        val_type_list = [
            'test_seen_obj_seen_act',
            'test_seen_obj_unseen_act',
            'test_unseen_obj_seen_act',
            'test_unseen_obj_unseen_act'
        ]
    else:
        val_type_list = [
            'val_seen_obj_seen_act',
            'val_seen_obj_unseen_act',
            'val_unseen_obj_seen_act',
            'val_unseen_obj_unseen_act'
        ]

    val_loader_dict = {}
    for val_type_i in val_type_list:
        dataset_configs = config['{}_dataset_args'.format(val_type_i)]
        dataset_configs['use_gt_bbox'] = config['use_gt_bbox']
        if 'comb_gt_det' in config.keys():
            dataset_configs['comb_gt_det'] = config['comb_gt_det']
        val_dataset_i = datasets.make(config['{}_dataset'.format(val_type_i)], **dataset_configs)
        if utils.is_main_process():
            utils.log('{} dataset: {} samples'.format(val_type_i, len(val_dataset_i)))

        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset_i)
        else:
            val_sampler = None
        val_loader_i = torch.utils.data.DataLoader(
            val_dataset_i,
            batch_size=args.batch_size_val,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=collate_images_boxes_dict
        )
        val_loader_dict[val_type_i] = val_loader_i

    ########

    #### Model and optimizer ####
    if args.load != None:
        print('loading pretrained model: ', args.load)
        checkpoint = torch.load(args.load)
        # state_dict = checkpoint['state_dict']  ### when the ptrtraining and training have the same gpu process

        ### if the pretraining use parallel gpu, training use onyl one, use the following state_dict
        from collections import OrderedDict
        state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:] # remove `module.`
            state_dict[name] = v
        
        # if config.get('load_encoder'):
        #     print('loading pretrained encoder: ', config['load_encoder'])
        #     pretrain = config.get('encoder_pretrain').lower()
        #     if pretrain != 'scratch':
        #         pretrain_encoder_path = config['load_encoder'].format(pretrain)
        model = models.load(state_dict, config['stud_model'], **config['stud_model_args'])
        ema_model = models.load(state_dict, config['stud_model'], **config['stud_model_args'])
    else:  
        args.start_epoch = 0
        checkpoint = None
        model = models.make(config['stud_model'], **config['stud_model_args'])
        ema_model = models.make(config['stud_model'], **config['stud_model_args'])
        if config.get('load_encoder'):
            print('loading pretrained encoder: ', config['load_encoder'])
            pretrain = config.get('encoder_pretrain').lower()
            if pretrain != 'scratch':
                pretrain_model_path = config['load_encoder'].format(pretrain)
                state_dict = torch.load(pretrain_model_path, map_location='cpu')
                missing_keys, unexpected_keys = model.encoder.encoder.load_state_dict(state_dict, strict=False)
                # for key in missing_keys:
                #     assert key.startswith('g_mlp.') \
                #         or key.startswith('proj') \
                #         or key.startswith('trans') \
                #         or key.startswith('roi_processor') \
                #         or key.startswith('roi_dim_processor') \
                #         or key.startswith('classifier'), key
                # for key in unexpected_keys:
                #     assert key.startswith('fc.')
                if utils.is_main_process():
                    utils.log('==> Successfully loaded {} for the enocder.'.format(pretrain_model_path))

        if args.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if utils.is_main_process():
            utils.log(model)

    if args.distributed:
        torch.cuda.set_device(gpu)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=False)
        ema_model = torch.nn.parallel.DistributedDataParallel(ema_model.cuda(), device_ids=[gpu], find_unused_parameters=False)
                
    else:
        model = torch.nn.DataParallel(model.cuda())
        ema_model = torch.nn.DataParallel(ema_model.cuda())
    
    ### init ema model
    # for param in ema_model.parameters():
    #     param.detach_()
    
    for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False  # not update by gradient


    if utils.is_main_process() and not args.test_only:
        utils.log('num params: {}'.format(utils.compute_n_params(model)))
        utils.log('Results will be saved to {}'.format(save_path))

    max_steps = min(len(train_loader), config['train_batches']) * config['max_epoch']
    optimizer, lr_scheduler, update_lr_every_epoch = utils.make_optimizer(
        model.parameters(),
        config['optimizer'], max_steps, **config['optimizer_args']
    )
    assert lr_scheduler is not None
    args.update_lr_every_epoch = update_lr_every_epoch
    
    if checkpoint is not None and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch']
    if args.test_only:
        filename = args.test_model
        print(filename)
        assert os.path.exists(filename)
        ckpt = torch.load(filename, map_location='cpu')
        start_epoch = ckpt['epoch']
        print("start_epoch", start_epoch)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        if utils.is_main_process():
            utils.log('==> Sucessfully resumed from a checkpoint {}, start_epoch{}'.format(filename, start_epoch))
    else:
        start_epoch = args.start_epoch

    ######## Training & Validation

    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    if args.test_only:
        test_acc = []
        test_acc_type = []
        for val_type_i, val_loader_i in val_loader_dict.items():
            loss_val_i, acc_val_i, output_list_i, logits_list_i, loss_ce_i, loss_cts_i, loss_subs_i, result_collect_all, result_collect_fail = validate(val_loader_i, model, 0, args)
            print("result_collect_all", result_collect_all)
            print("result_collect_fail", result_collect_fail)
            result_collect_ratio = {}
            for i in result_collect_all.keys():
                result_collect_ratio[i] = result_collect_fail[i].item()/result_collect_all[i]
            print("result_collect_ratio", result_collect_ratio)

            test_acc.append(acc_val_i)
            test_acc_type.append(val_type_i)
            if utils.is_main_process():
                print('testing result: ', val_type_i, acc_val_i)
                utils.log('{} test_result: loss {:.4f}, acc: {:.4f},  loss_ce {:.4f},  loss_cts {:.4f}, loss_subspace {:.4f}.'.format(val_type_i, loss_val_i, acc_val_i, loss_ce_i, loss_cts_i, loss_subs_i))
                utils.log('{} output_specific_false-results: {}, logits: {}'.format(val_type_i, output_list_i, logits_list_i))
        print('summary')
        for i, j in zip(test_acc_type, test_acc):
            print(i, j)
        print('avg', sum(test_acc)/4)
        if utils.is_main_process():
            logger.close()
        return 0

    best_val_result = {type: 0.0 for type in val_loader_dict.keys()}
    for epoch in range(start_epoch, max_epoch):

        epoch_log = epoch + 1
        if args.distributed:
            train_sampler.set_epoch(epoch)
        args.weight_ce = min(0.3, 0.05*1.1**(epoch+1))
        loss_train, acc_train, _, loss_ce_train, loss_rotce_train, loss_mse_train = train(train_loader, model, ema_model, optimizer, lr_scheduler, epoch_log, args)
        filename = os.path.join(save_path, 'final_model.pth')
        ckpt = {
            'epoch': epoch_log,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }
        torch.save(ckpt, filename)

        
        if args.update_lr_every_epoch:
            lr_scheduler.step()
        if utils.is_main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('acc_train', acc_train, epoch_log)
            writer.add_scalar('loss_ce_train', loss_ce_train, epoch_log)
            writer.add_scalar('loss_rotce_train', loss_rotce_train, epoch_log)
            writer.add_scalar('loss_mse_train', loss_mse_train, epoch_log)
            # for name, param in model.named_parameters():
            #     writer.add_histogram(name, param)

        if epoch_log % config['eval_epoch'] == 0:
            avg_acc_val = 0
            for val_type_i, val_loader_i in val_loader_dict.items():
                loss_val_i, acc_val_i, _, _, loss_ce_i, loss_cts_i, loss_subs_i, _, _ = validate(val_loader_i, model, epoch_log, args)
                if acc_val_i > best_val_result[val_type_i]:
                    best_val_result[val_type_i] = acc_val_i
                if utils.is_main_process():
                    utils.log('{} result: loss {:.4f}, acc: {:.4f}.'.format(val_type_i, loss_val_i, acc_val_i))
                    writer.add_scalar('loss_{}'.format(val_type_i), loss_val_i, epoch_log)
                    writer.add_scalar('loss_ce_{}'.format(val_type_i), loss_ce_i, epoch_log)
                    writer.add_scalar('loss_cts_like_{}'.format(val_type_i), loss_cts_i, epoch_log)
                    writer.add_scalar('loss_subsapce_{}'.format(val_type_i), loss_subs_i, epoch_log)
                    writer.add_scalar('acc_{}'.format(val_type_i), acc_val_i, epoch_log)
                avg_acc_val += acc_val_i
            avg_acc_val /= len(val_loader_dict.keys())

        utils.log('Best val results so far:')
        utils.log(best_val_result)

        if avg_acc_val > max_va and utils.is_main_process():
            max_va = avg_acc_val
            filename = os.path.join(save_path, 'best_model.pth')
            ckpt = {
                'epoch': epoch_log,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }
            torch.save(ckpt, filename)
        if utils.is_main_process():
            writer.flush()

    if utils.is_main_process():
        logger.close()

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    # for ema_param, param in zip(ema_model.parameters(), model.parameters()):
    #     ema_param.data.mul_(alpha).add_(param.data, alpha = 1 - alpha)
    with torch.no_grad():
        model_state_dict = model.state_dict()
        ema_model_state_dict = ema_model.state_dict()
        for entry in ema_model_state_dict.keys():
            ema_param = ema_model_state_dict[entry].clone().detach()
            param = model_state_dict[entry].clone().detach()
            new_param = (ema_param * alpha) + (param * (1. - alpha))
            ema_model_state_dict[entry] = new_param
        ema_model.load_state_dict(ema_model_state_dict)

def train(train_loader, model, ema_model, optimizer, lr_scheduler, epoch, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    main_loss_meter = utils.AverageMeter()
    aux_loss_meter = utils.AverageMeter()
    loss_meter = utils.AverageMeter()
    acc_meter = utils.AverageMeter()
    intersection_meter = utils.AverageMeter()
    union_meter = utils.AverageMeter()
    target_meter = utils.AverageMeter()
    loss_ce_meter = utils.AverageMeter()
    loss_cts_meter = utils.AverageMeter()
    loss_subspace_meter = utils.AverageMeter()
    loss_rotce_meter = utils.AverageMeter()
    loss_mse_meter = utils.AverageMeter()
    
    global global_step
    logits_list = []
    config = args.config

    # train
    model.train()
    ema_model.train()

    if utils.is_main_process():
        args.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    lr = optimizer.param_groups[0]['lr']

    # temp_pos = torch.as_tensor([1])
    # temp_neg = torch.as_tensor([1])
    end = time.time()
    max_iter = config['max_epoch'] * len(train_loader)
    if global_step != 0:
        update_ema_variables(model, ema_model, 0.99, global_step)

    for batch_idx, data_dict in enumerate(train_loader):
        
        if batch_idx >= config['train_batches']:
            break

        x_shot = data_dict['shot_ims'].cuda(non_blocking=True)
        teach_ims = data_dict['teach_ims'].cuda(non_blocking=True)

        x_query = data_dict['query_ims'].cuda(non_blocking=True)
        rot_query_ims = data_dict['rot_query_ims'].cuda(non_blocking=True)
        
        label_query = data_dict['query_labs'].cuda(non_blocking=True).view(-1)
        if 'shot_boxes' in data_dict:
            assert 'query_boxes' in data_dict
            assert 'shot_boxes_dim' in data_dict
            assert 'query_boxes_dim' in data_dict
            ### obj bboxes
            shot_boxes = data_dict['shot_boxes']
            teach_boxes = data_dict['teach_boxes']
            for idx, shot_boxes_i in enumerate(shot_boxes):
                shot_boxes_i = [Boxes(boxes.tensor.cuda(non_blocking=True)) for boxes in shot_boxes_i]
                shot_boxes[idx] = shot_boxes_i

            for idx, shot_boxes_i in enumerate(teach_boxes):
                shot_boxes_i = [Boxes(boxes.tensor.cuda(non_blocking=True)) for boxes in shot_boxes_i]
                teach_boxes[idx] = shot_boxes_i

            query_boxes = data_dict['query_boxes']
            rot_query_box = data_dict['rot_query_box']
            for idx, query_boxes_i in enumerate(query_boxes):
                query_boxes_i = [Boxes(boxes.tensor.cuda(non_blocking=True)) for boxes in query_boxes_i]
                query_boxes[idx] = query_boxes_i
            for idx, query_boxes_i in enumerate(rot_query_box):
                query_boxes_i = [Boxes(boxes.tensor.cuda(non_blocking=True)) for boxes in query_boxes_i]
                rot_query_box[idx] = query_boxes_i            
            
            ### human roi
            shot_human_roi = data_dict['shot_human_roi']
            teach_hum_roi = data_dict['teach_hum_roi']
            if shot_human_roi[0][0] is not None:
                for idx, shot_human_roi_i in enumerate(shot_human_roi):
                    shot_human_roi_i = [Boxes(boxes.tensor.cuda(non_blocking=True)) for boxes in shot_human_roi_i]
                    shot_human_roi[idx] = shot_human_roi_i
                for idx, shot_human_roi_i in enumerate(teach_hum_roi):
                    shot_human_roi_i = [Boxes(boxes.tensor.cuda(non_blocking=True)) for boxes in shot_human_roi_i]
                    teach_hum_roi[idx] = shot_human_roi_i
            else:
                teach_hum_roi = shot_human_roi

            query_human_roi = data_dict['query_human_roi']
            rot_query_humroi = data_dict['rot_query_humroi']
            if query_human_roi[0][0] is not None:
                for idx, query_human_roi_i in enumerate(query_human_roi):
                    query_human_roi_i = [Boxes(boxes.tensor.cuda(non_blocking=True)) for boxes in query_human_roi_i]
                    query_human_roi[idx] = query_human_roi_i    
                for idx, query_human_roi_i in enumerate(rot_query_humroi):
                    query_human_roi_i = [Boxes(boxes.tensor.cuda(non_blocking=True)) for boxes in query_human_roi_i]
                    rot_query_humroi[idx] = query_human_roi_i  
            else:
                rot_query_humroi = query_human_roi

        torch.cuda.empty_cache()

        data_time.update(time.time() - end)

        count_iter = ((epoch-1)*len(train_loader)+batch_idx+1)
        with torch.cuda.amp.autocast(enabled=args.amp):
            # if shot_boxes is not None and query_boxes is not None:
            if count_iter> 1000:
                logits, subspace_loss, cts_like_loss, rot_q_logits = model(
                    x_shot,
                    x_query,
                    shot_boxes=shot_boxes,
                    query_boxes=query_boxes,
                    shot_human_roi = shot_human_roi,
                    query_human_roi = query_human_roi,
                    roi_position = args.no_roi_position,
                    cts_loss_weight = args.config['cts_lambda'],
                    # ema = False,
                    rot_query_ims = rot_query_ims,
                    rot_query_box = rot_query_box,
                    rot_query_humroi = rot_query_humroi,
                )                    
                logits = logits.view(-1, args.n_train_way)
                rot_q_logits = rot_q_logits.view(-1, args.n_train_way)

                rot_q_logits_ema, _, _, _ = ema_model(
                    teach_ims,
                    rot_query_ims.clone(),
                    shot_boxes=teach_boxes,
                    query_boxes=deepcopy(rot_query_box),
                    shot_human_roi = teach_hum_roi,
                    query_human_roi = deepcopy(rot_query_humroi),
                    roi_position = args.no_roi_position,
                    cts_loss_weight = args.config['cts_lambda'],
                    # teach_ims = teach_ims,
                    # teach_boxes=teach_boxes,
                    # teach_hum_roi=teach_hum_roi,
                    # ema = True,
                    # rot_query_ims = rot_query_ims,
                    # rot_query_box = rot_query_box,
                    # rot_query_humroi = rot_query_humroi
                )                    
                # ema_logits = ema_logits.view(-1, args.n_train_way)
                rot_q_logits_ema = rot_q_logits_ema.view(-1, args.n_train_way)
            else:
                logits, subspace_loss, cts_like_loss, rot_q_logits = model(
                    teach_ims,
                    x_query,
                    shot_boxes=teach_boxes,
                    query_boxes=query_boxes,
                    shot_human_roi = teach_hum_roi,
                    query_human_roi = query_human_roi,
                    roi_position = args.no_roi_position,
                    cts_loss_weight = args.config['cts_lambda'],
                    # ema = False,
                    rot_query_ims = rot_query_ims,
                    rot_query_box = rot_query_box,
                    rot_query_humroi = rot_query_humroi,
                )                    
                logits = logits.view(-1, args.n_train_way)
                rot_q_logits = rot_q_logits.view(-1, args.n_train_way)

                rot_q_logits_ema, _, _, _ = ema_model(
                    teach_ims.clone(),
                    rot_query_ims.clone(),
                    shot_boxes=deepcopy(teach_boxes),
                    query_boxes=deepcopy(rot_query_box),
                    shot_human_roi = deepcopy(teach_hum_roi),
                    query_human_roi = deepcopy(rot_query_humroi),
                    roi_position = args.no_roi_position,
                    cts_loss_weight = args.config['cts_lambda'],
                    # teach_ims = teach_ims,
                    # teach_boxes=teach_boxes,
                    # teach_hum_roi=teach_hum_roi,
                    # ema = True,
                    # rot_query_ims = rot_query_ims,
                    # rot_query_box = rot_query_box,
                    # rot_query_humroi = rot_query_humroi
                )                    
                # ema_logits = ema_logits.view(-1, args.n_train_way)
                rot_q_logits_ema = rot_q_logits_ema.view(-1, args.n_train_way)                
            
            
            
            rot_q_logits_ema = Variable(rot_q_logits_ema.detach().data, requires_grad=False)
            
            loss_ce = F.cross_entropy(logits.float(), label_query)

            # bool_teach = torch.as_tensor([True, False]).to(logits.device)
            bool_teach = torch.max(F.softmax(rot_q_logits_ema.float(), dim=1), dim = 1)[0] > 0.95
            count = -1
            # loss_mse = torch.as_tensor(1E-6).to(logits.device)
            loss_rotce = torch.as_tensor(1E-6).to(logits.device)
            # print("!!!!!!!", rot_q_logits[count], rot_q_logits_ema[count])
            # print("--------------------", F.cross_entropy(rot_q_logits[count],  F.softmax(rot_q_logits_ema[count].float())) , F.softmax(rot_q_logits_ema[count].float()))
            for i in bool_teach:
                if i == True:
                    count += 1
                    loss_rotce = loss_rotce + F.cross_entropy(rot_q_logits[count],  (rot_q_logits_ema[count].float())) 
                    # loss_rotce = loss_rotce + F.cross_entropy(rot_q_logits[count],  rot_q_logits_ema[count]) 
                    # loss_mse = loss_mse + F.mse_loss(F.softmax(rot_q_logits[count].unsqueeze(0).float(), dim=1), F.softmax(rot_q_logits_ema[count].unsqueeze(0).float(), dim=1), size_average=False) 

            
            if loss_rotce.item() <0.01:
                # loss = loss_ce * 1E-6
                # print("original train")
                weight_loss_orig =  1/(1+math.exp(count_iter/3000-2))
                loss = weight_loss_orig*(loss_ce + args.config['subspace_loss_lambda'] * subspace_loss + args.config['cts_lambda'] * cts_like_loss  )
            else:
                # print("teacher: ", bool_teach, data_dict['task_name'], data_dict['neg_query_act'])

                weight_loss_orig =  1/(1+math.exp(count_iter/3000-2))
                weight_loss_rotce = 0.2/(1+math.exp(-count_iter/3000+2)) if count_iter > 1000 else 1E-6

                loss_rotce = loss_rotce / (count+1)
                # loss_mse = loss_mse / (count+1)
                loss = weight_loss_rotce*loss_rotce + weight_loss_orig*(loss_ce + args.config['subspace_loss_lambda'] * subspace_loss + args.config['cts_lambda'] * cts_like_loss )
                # print("teacher loss", weight_loss_rotce, loss_rotce, "/ loss ce: ", weight_loss_orig )
            acc, _ = utils.compute_acc(logits, label_query)
        
        del x_shot, x_query, teach_ims, rot_query_ims
        torch.cuda.empty_cache()

        optimizer.zero_grad()
        # print("before backward", logits.device)
        
        loss.backward()
        # print("after backward", logits.device)
        optimizer.step()


        global_step += 1
        


        lrs = lr_scheduler.get_last_lr()
        if not args.update_lr_every_epoch:
            lr_scheduler.step()

        time.sleep(1E-2)
        n = logits.size(0)
        if args.multiprocessing_distributed:
            loss = loss * n  
            cts_like_loss = cts_like_loss * n
            subspace_loss = subspace_loss * n
            loss_ce = loss_ce * n
            # loss_mse = loss_mse * int(sum(bool_teach))
            loss_rotce = loss_rotce * int(sum(bool_teach))
            acc = acc * n
            count = label_query.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss)
            dist.all_reduce(acc)
            dist.all_reduce(count)
            dist.all_reduce(loss_ce)
            dist.all_reduce(cts_like_loss)
            dist.all_reduce(subspace_loss)
            # dist.all_reduce(loss_mse)
            dist.all_reduce(loss_rotce)
            n = count.item()
            loss = loss / n
            cts_like_loss = cts_like_loss / n
            subspace_loss = subspace_loss / n
            loss_ce = loss_ce / n 
            # loss_mse = loss_mse / max(int(sum(bool_teach)), 1 )
            loss_rotce = loss_rotce /  max(int(sum(bool_teach)), 1 )
            acc = acc / n
        

        logits_list.append(logits)
        loss_meter.update(loss.item(), logits.size(0))
        loss_ce_meter.update(loss_ce.item(), logits.size(0))
        loss_cts_meter.update(cts_like_loss.item(), logits.size(0))
        loss_subspace_meter.update(subspace_loss.item(), logits.size(0))
        loss_rotce_meter.update(loss_rotce.item(), max(int(sum(bool_teach)), 1 ) )
        # loss_mse_meter.update(loss_mse.item(), max(int(sum(bool_teach)), 1 )  )

        acc_meter.update(acc.item(), logits.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + batch_idx + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (batch_idx + 1) % config['print_freq'] == 0 and utils.is_main_process():
            utils.log(
                'Epoch: [{}/{}][{}/{}] '
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                'Remain {remain_time} '
                'Loss {loss_meter.val:.4f} '
                'Acc {acc_meter.val:.4f} '
                'lr {lr:.6f}'.format(
                    epoch, config['max_epoch'], batch_idx + 1, len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    remain_time=remain_time,
                    loss_meter=loss_meter,
                    acc_meter=acc_meter,
                    lr=lrs[0]
                )
            )
            print('Loss ce: ', loss_ce_meter.val,
                'loss_rotce: ', loss_rotce_meter.val,
                'Loss cts: ', loss_cts_meter.val,
                'Loss subspace * 0.03: ', 0.03*loss_subspace_meter.val,
                'logits one ', logits[0],)
    if utils.is_main_process():
        utils.log('Train result at epoch [{}/{}]: loss {:.4f}, acc {:.4f}.'.format(epoch, config['max_epoch'], loss_meter.avg, acc_meter.avg))
    return loss_meter.avg, acc_meter.avg, logits_list, loss_ce_meter.avg, loss_rotce_meter.avg, loss_mse_meter.avg


def validate(val_loader, model, epoch_log, args):
    # eval
    model.eval()
    logits_list = []
    config = args.config
    loss_meter = utils.AverageMeter()
    acc_meter = utils.AverageMeter()
    loss_ce_meter = utils.AverageMeter()
    loss_cts_meter = utils.AverageMeter()
    loss_subspace_meter = utils.AverageMeter()
    output_list = []
    result_collect_all = {}
    result_collect_fail = {}
    np.random.seed(0)
    for data_dict in tqdm(val_loader):
        x_shot = data_dict['shot_ims'].cuda(non_blocking=True)
        x_query = data_dict['query_ims'].cuda(non_blocking=True)
        label_query = data_dict['query_labs'].cuda(non_blocking=True).view(-1)
        if 'shot_boxes' in data_dict:
            assert 'query_boxes' in data_dict
            shot_boxes = data_dict['shot_boxes']
            for idx, shot_boxes_i in enumerate(shot_boxes):
                shot_boxes_i = [Boxes(boxes.tensor.cuda(non_blocking=True)) for boxes in shot_boxes_i]
                shot_boxes[idx] = shot_boxes_i

            query_boxes = data_dict['query_boxes']
            for idx, query_boxes_i in enumerate(query_boxes):
                query_boxes_i = [Boxes(boxes.tensor.cuda(non_blocking=True)) for boxes in query_boxes_i]
                query_boxes[idx] = query_boxes_i
            
            ### human roi
            shot_human_roi = data_dict['shot_human_roi']
            if shot_human_roi[0] is not None:
                for idx, shot_human_roi_i in enumerate(shot_human_roi):
                    shot_human_roi_i = [Boxes(boxes.tensor.cuda(non_blocking=True)) for boxes in shot_human_roi_i]
                    shot_human_roi[idx] = shot_human_roi_i

            query_human_roi = data_dict['query_human_roi']
            if query_human_roi[0] is not None:
                for idx, query_human_roi_i in enumerate(query_human_roi):
                    query_human_roi_i = [Boxes(boxes.tensor.cuda(non_blocking=True)) for boxes in query_human_roi_i]
                    query_human_roi[idx] = query_human_roi_i  

            assert 'shot_boxes_dim' in data_dict
            assert 'query_boxes_dim' in data_dict
            shot_boxes_dim = data_dict['shot_boxes_dim'].cuda(non_blocking=True)
            query_boxes_dim = data_dict['query_boxes_dim'].cuda(non_blocking=True)
        else:
            shot_boxes = None
            query_boxes = None
            shot_boxes_dim = None
            query_boxes_dim = None
            shot_human_roi = None
            query_human_roi = None


        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.amp):
                if shot_boxes is not None and query_boxes is not None:
                    logits, subspace_loss, cts_like_loss, _ = model(
                        x_shot,
                        x_query,
                        shot_boxes=shot_boxes,
                        query_boxes=query_boxes,
                        shot_human_roi = shot_human_roi,
                        query_human_roi = query_human_roi,
                        roi_position = args.no_roi_position,
                        shot_boxes_dim=shot_boxes_dim,
                        query_boxes_dim=query_boxes_dim,
                        cts_loss_weight =args.config['cts_lambda'] ,
                        eval=True
                    )
                    logits = logits.view(-1, args.n_train_way)
                else:
                    logits, subspace_loss, cts_like_loss = model(x_shot, x_query, eval=True)
                    logits = logits.view(-1, args.n_train_way)

                loss_ce = F.cross_entropy(logits.float(), label_query)
                loss = loss_ce + args.config['subspace_loss_lambda'] * subspace_loss
                + args.config['cts_lambda'] * cts_like_loss   
                acc, output = utils.compute_acc(logits, label_query)
                
        # print("-----------------------",data_dict['flag'])
        if data_dict['flag'][0] not in result_collect_all.keys():
            result_collect_all[data_dict['flag'][0]] = 2
        else:
            result_collect_all[data_dict['flag'][0]] += 2
        if data_dict['flag'][0] not in result_collect_fail.keys():
            result_collect_fail[data_dict['flag'][0]] = 2*(1-acc)
        else:
            result_collect_fail[data_dict['flag'][0]] += 2*(1-acc)
        del x_shot, x_query
        torch.cuda.empty_cache()

        logits_list.append(logits)
        n = logits.size(0)
        if args.multiprocessing_distributed:
            loss = loss * n  # not considering ignore pixels
            acc = acc * n
            cts_like_loss = cts_like_loss * n
            subspace_loss = subspace_loss * n
            loss_ce = loss_ce * n
            count = logits.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss)
            dist.all_reduce(acc)
            dist.all_reduce(count)
            dist.all_reduce(loss_ce)
            dist.all_reduce(cts_like_loss)
            dist.all_reduce(subspace_loss)
            n = count.item()
            loss = loss / n
            acc = acc / n
            cts_like_loss = cts_like_loss / n
            subspace_loss = subspace_loss / n
            loss_ce = loss_ce / n  
            output_list.extend(output.tolist())
        else:
            loss = torch.mean(loss)
            loss_ce = torch.mean(loss_ce)
            cts_like_loss = torch.mean(cts_like_loss.float())
            subspace_loss = torch.mean(subspace_loss)
            acc = torch.mean(acc)
            output_list.extend(output.tolist()) ## 1 means right, 0 means false

        out_false_ind = [index for (index,value) in enumerate(output_list) if value == 0]
        loss_meter.update(loss.item(), logits.size(0))
        loss_ce_meter.update(loss_ce.item(), logits.size(0))
        loss_cts_meter.update(cts_like_loss.item(), logits.size(0))
        loss_subspace_meter.update(subspace_loss.item(), logits.size(0))
        acc_meter.update(acc.item(), logits.size(0))
    return loss_meter.avg, acc_meter.avg, out_false_ind, logits_list, loss_ce_meter.avg, loss_cts_meter.avg, loss_subspace_meter.avg, result_collect_all, result_collect_fail

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', default = "configs/my_metric.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--save_dir', default='./save_dist')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--train_gpu', default='0, 1')
    parser.add_argument('--split_gpu', action = 'store_true')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--test_model', default=None)
    parser.add_argument('--no_roi_position', action = 'store_false')  ## input of roialign is 1*1 or not (True means not 1*1)
    # distributed training
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--dist-backend', default='nccl')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--load', default = None)
    parser.add_argument('--stage', default = 1)
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}",
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    args.multiprocessing_distributed = True

    config = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)
    if args.opts is not None:
        config = utils.override_cfg_from_list(config, args.opts)
    print('config:')
    print(config)
    main(config)
