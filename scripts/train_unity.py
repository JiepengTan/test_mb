import os
import random
import copy
import time
import sys
import shutil
import argparse
import errno
import subprocess
import math
import numpy as np
from collections import defaultdict, OrderedDict
import tensorboardX
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from lib.utils.tools import *
from lib.model.loss import *
from lib.model.loss_unity import *
from lib.utils.utils_mesh import *
from lib.utils.utils_smpl import *
from lib.utils.utils_data import *
from lib.utils.learning import *
from lib.data.dataset_unity import UnityDataset3D
from lib.model.model_unity import UnityRegressor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-freq', '--print_freq', default=100)
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('--epoch_script', type=str, default="infer_unity.sh",  help='execute after echo epoch ')
    parser.add_argument('--debug', type=int, default=0,  help='is debug mode ')
    opts = parser.parse_args()
    return opts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def validate(test_loader, model, criterion, dataset_name='unity'):
    model.eval()
    print(f'===========> validating {dataset_name}')
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_dict = {'loss_3d_pos': AverageMeter(), 
                   'loss_a': AverageMeter(), 
                   'loss_a_up': AverageMeter(), 
                   'loss_av': AverageMeter(), 
                   'loss_norm': AverageMeter(),
    }
    mpjpes = AverageMeter()
    results = defaultdict(list)
    with torch.no_grad():
        end = time.time()
        for idx, (batch_input, batch_gt) in tqdm(enumerate(test_loader)):
            batch_size, clip_len = batch_input.shape[:2]
            if torch.cuda.is_available():
                batch_gt['theta'] = batch_gt['theta'].cuda().float()
                batch_gt['kp_3d'] = batch_gt['kp_3d'].cuda().float()
                batch_gt['dir_fu'] = batch_gt['dir_fu'].cuda().float()
                batch_input = batch_input.cuda().float()
            output = model(batch_input)    
            loss_dict = criterion(output, batch_gt)
            loss = args.lambda_3d      * loss_dict['loss_3d_pos']      + \
                   args.lambda_a       * loss_dict['loss_a']           + \
                   args.lambda_a_up    * loss_dict['loss_a_up']        + \
                   args.lambda_av      * loss_dict['loss_av']          + \
                   args.lambda_norm    * loss_dict['loss_norm'] 

            # update metric
            losses.update(loss.item(), batch_size)
            loss_str = ''
            for k, v in loss_dict.items():
                losses_dict[k].update(v.item(), batch_size)
                loss_str += '{0} {loss.val:.3f} ({loss.avg:.3f})\t'.format(k, loss=losses_dict[k])

            for keys in output[0].keys():
                output[0][keys] = output[0][keys].detach().cpu().numpy()
                batch_gt[keys] = batch_gt[keys].detach().cpu().numpy()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % int(opts.print_freq) == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      ''.format(
                       idx, len(test_loader), loss_str, batch_time=batch_time,loss=losses))

    print(f'=======================> {dataset_name} validation done: ', loss_str)
    return losses.avg,losses_dict


def train_epoch(args, opts, model, train_loader, losses_train, losses_dict, mpjpes, criterion, optimizer, batch_time, data_time, epoch):
    model.train()
    end = time.time()
    for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):
        data_time.update(time.time() - end)
        batch_size = len(batch_input)

        if torch.cuda.is_available():
            batch_gt['theta'] = batch_gt['theta'].cuda().float()
            batch_gt['kp_3d'] = batch_gt['kp_3d'].cuda().float()
            batch_gt['dir_fu'] = batch_gt['dir_fu'].cuda().float()
            batch_input = batch_input.cuda().float()
        output = model(batch_input)
        optimizer.zero_grad()
        loss_dict = criterion(output, batch_gt)
        loss = args.lambda_3d       * loss_dict['loss_3d_pos']      + \
                args.lambda_a       * loss_dict['loss_a']           + \
                args.lambda_a_up    * loss_dict['loss_a_up']        + \
                args.lambda_av      * loss_dict['loss_av']          + \
                args.lambda_norm    * loss_dict['loss_norm'] 
        
        losses_train.update(loss.item(), batch_size)
        loss_str = ''
        for k, v in loss_dict.items():
            losses_dict[k].update(v.item(), batch_size)
            loss_str += '{0} {loss.val:.3f} ({loss.avg:.3f})\t'.format(k, loss=losses_dict[k])
        
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % int(opts.print_freq) == 0:
            print('Train: [{0}][{1}/{2}]\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                ''.format(
                epoch, idx + 1, len(train_loader), loss_str, loss=losses_train))
            sys.stdout.flush()

def save_checkpoint(chk_path, epoch, lr, optimizer, model, best_jpe):
    print('Saving checkpoint to', chk_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'best_jpe' : best_jpe
    }, chk_path)

def train_with_config(args, opts):
    opts.debug = opts.debug != 0
    if(opts.debug):
        args.batch_size = 1
    print(opts)
    print(args)
    try:
        os.makedirs(opts.checkpoint)
        shutil.copy(opts.config, opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))
    model_backbone = load_backbone(args)
    if args.finetune:
        if opts.resume or opts.evaluate:
            pass
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading backbone', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_pos']
            model_backbone = load_pretrained_weights(model_backbone, checkpoint)
    if args.partial_train:
        model_backbone = partial_train_layers(model_backbone, args.partial_train)
    #  theta  (N, T, 24*3)
    #  kp_3d  (N, T, 17, 3)
    #  dir_fu (N, T, 24*6) #target dirs forward ,up
    model = UnityRegressor(args, backbone=model_backbone, dim_rep=args.dim_rep, hidden_dim=args.hidden_dim, dropout_ratio=args.dropout, num_joints=args.num_joints)
    criterion = UnityLoss(loss_type = args.loss_type)
    best_jpe = 9999.0
    model_params = 0
    for parameter in model.parameters():
        if parameter.requires_grad == True:
            model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
    print('Loading dataset...')
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }

    train_dataset = UnityDataset3D(args, args.subset_list, 'train', opts.debug)
    test_dataset = UnityDataset3D(args, args.subset_list, 'test', opts.debug)
    
    train_loader = DataLoader(train_dataset, **trainloader_params)
    test_loader = DataLoader(test_dataset, **testloader_params)
    print('INFO: Training on {} batches (unity)'.format(len(train_loader)))
    
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        
    chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
    if os.path.exists(chk_filename):
        opts.resume = chk_filename
    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)
    if opts.evaluate:
        print("no implement!!")
    else: 
        optimizer = optim.AdamW(
                [     {"params": filter(lambda p: p.requires_grad, model.module.backbone.parameters()), "lr": args.lr_backbone},
                      {"params": filter(lambda p: p.requires_grad, model.module.head.parameters()), "lr": args.lr_head},
                ],      lr=args.lr_backbone, 
                        weight_decay=args.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
        st = 0
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint['lr']
            if 'best_jpe' in checkpoint and checkpoint['best_jpe'] is not None:
                best_jpe = checkpoint['best_jpe']
        
        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            losses_train = AverageMeter()
            losses_dict = {
                'loss_3d_pos': AverageMeter(), 
                'loss_a': AverageMeter(), 
                'loss_a_up': AverageMeter(), 
                'loss_av': AverageMeter(), 
                'loss_norm': AverageMeter(),
            }
            mpjpes = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            
            train_epoch(args, opts, model, train_loader, losses_train, losses_dict, mpjpes, criterion, optimizer, batch_time, data_time, epoch)
            test_loss, test_losses_dict = validate(test_loader, model, criterion, 'unity')

            train_writer.add_scalar('test_loss', test_loss, epoch + 1)
            for k, v in test_losses_dict.items():
                train_writer.add_scalar('test_loss/'+k, v.avg, epoch + 1)

            
            train_writer.add_scalar('train_loss', losses_train.avg, epoch + 1)
            for k, v in losses_dict.items():
                train_writer.add_scalar('train_loss/'+k, v.avg, epoch + 1)
                
            # Decay learning rate exponentially
            scheduler.step()

            test_mpjpe = test_losses_dict['loss_a'].avg # TODO
            #print(f"test_mpjpe ==  {test_mpjpe}")

            lr = scheduler.get_last_lr()
            best_jpe_cur = test_mpjpe

            # Save latest checkpoint.
            chk_path = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            save_checkpoint(chk_path, epoch, lr, optimizer, model, best_jpe)
            
            # Save checkpoint if necessary.
            if (epoch+1) % args.checkpoint_frequency == 0:
                chk_path = os.path.join(opts.checkpoint, 'epoch_{}.bin'.format(epoch))
                save_checkpoint(chk_path, epoch, lr, optimizer, model, best_jpe)

            # Save best checkpoint.
            best_chk_path = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))
            if best_jpe_cur < best_jpe:
                best_jpe = best_jpe_cur
                save_checkpoint(best_chk_path, epoch, lr, optimizer, model, best_jpe)


            # execute script 
            if opts.epoch_script != "":
                script_path = os.path.join(os.getcwd(),opts.epoch_script)
                print(f"======= run scirpt {script_path} {epoch} ======= ")
                process = subprocess.Popen([script_path, str(epoch)])

if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train_with_config(args, opts)