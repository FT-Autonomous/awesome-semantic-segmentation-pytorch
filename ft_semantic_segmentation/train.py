import time
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


def show_image(img):
    plt.imshow(img)
    plt.show()

import argparse
import time
import datetime
import os
import shutil
import sys
import re

import wandb

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from .utils.visualize import mask_to_color
from .data.dataloader import get_segmentation_dataset, get_segmentation_dataset_names
from .models.model_zoo import get_segmentation_model, get_segmentation_model_names
from .utils.loss import get_segmentation_loss
from .utils.distributed import *
from .utils.logger import setup_logger
from .utils.lr_scheduler import WarmupPolyLR
from .utils.score import SegmentationMetric

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='fcn',
                        choices=get_segmentation_model_names(),
                        help='model name (default: fcn32s)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet18', 'resnet50', 'resnet101', 'resnet152', 
                            'densenet121', 'densenet161', 'densenet169', 'densenet201'],
                        help='backbone name (default: vgg16)')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        choices=get_segmentation_dataset_names(),
                        help='dataset name (default: pascal_voc)')
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--workers', '-j', type=int, default=4,
                        metavar='N', help='dataloader threads')
    # training hyper params
    parser.add_argument('--jpu', action='store_true', default=False,
                        help='JPU')
    parser.add_argument('--use-ohem', type=bool, default=False,
                        help='OHEM Loss for cityscapes dataset')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--warmup-iters', type=int, default=0,
                        help='warmup iters')
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3,
                        help='lr = warmup_factor * lr')
    parser.add_argument('--warmup-method', type=str, default='linear',
                        help='method of warmup')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='~/.torch/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--project', type=str,
                        help='wandb project name')
    parser.add_argument('--no-wandb', action='store_true',
                        help='make wandb commands behave as noops')
    args = parser.parse_args()

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'pascal_aug': 80,
            'pascal_voc': 50,
            'pcontext': 80,
            'ade20k': 160,
            'citys': 120,
            'sbu': 160,
        }
        args.epochs = epoches[args.dataset.lower()]
    if args.lr is None:
        lrs = {
            'coco': 0.004,
            'pascal_aug': 0.001,
            'pascal_voc': 0.0001,
            'pcontext': 0.001,
            'ade20k': 0.01,
            'citys': 0.01,
            'sbu': 0.001,
        }
        args.lr = lrs[args.dataset.lower()] / 8 * args.batch_size

    
    if args.no_wandb:
        wandb.init(mode="disabled")
    elif args.project:
        wandb.init(config=args, project=args.project)
    else:
        wandb.init()
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        train_dataset = get_segmentation_dataset(args.dataset, split='train', mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)
        args.iters_per_epoch = len(train_dataset) // (args.num_gpus * args.batch_size)
        args.max_iters = args.epochs * args.iters_per_epoch

        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iters)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(pretrained_base=False, model=args.model, dataset=args.dataset, backbone=args.backbone,
                                            aux=args.aux, jpu=args.jpu, norm_layer=BatchNorm2d).to(self.device)

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print(f'Resuming training, loading {args.resume}...')
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        self.criterion = get_segmentation_loss(args.model, nclass=train_dataset.num_class, use_ohem=args.use_ohem, aux=args.aux,
                                               aux_weight=args.aux_weight, ignore_index=-1).to(self.device)

        # optimizer, for model just includes pretrained, head and auxlayer
        params_list = list()
        if hasattr(self.model, 'pretrained'):
            params_list.append({'params': self.model.pretrained.parameters(), 'lr': args.lr})
        if hasattr(self.model, 'exclusive'):
            for module in self.model.exclusive:
                params_list.append({'params': getattr(self.model, module).parameters(), 'lr': args.lr * 10})
        self.optimizer = torch.optim.SGD(params_list,
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # lr scheduling
        self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                         max_iters=args.max_iters,
                                         power=0.9,
                                         warmup_factor=args.warmup_factor,
                                         warmup_iters=args.warmup_iters,
                                         warmup_method=args.warmup_method)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
        
        
        
                                                             output_device=args.local_rank)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)

        self.best_pred = 0.0

    def train(self):
        save_to_disk = get_rank() == 0
        epochs, max_iters = self.args.epochs, self.args.max_iters
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.args.iters_per_epoch
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        start_time = time.time()
        logger.info(f'Start training, Total Epochs: {epochs:d}, Total Iterations {max_iters:d}')

        self.model.train()
        wandb.watch(self.model)
        
        table = wandb.Table(columns=['Image', 'Actual Mask', 'Predicted Mask'])
        
        for iteration, (images, targets, _) in enumerate(self.train_loader):
            
            iteration = iteration + 1

            images = images.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(images)
            
            loss_dict = self.criterion(outputs, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            cost_time_string = str(datetime.timedelta(seconds=int(time.time() - start_time)))

            if iteration % log_per_iters == 0 and save_to_disk:
                logger.info(f"Iters: {iteration:d}/{max_iters:d}"
                            f" || Lr: {self.optimizer.param_groups[0]['lr']:.6f}"
                            f" || Loss: {losses_reduced:.4f}"
                            f" || Cost Time: {cost_time_string}"
                            f" || Estimated Time: {eta_string}")
                
            if not self.args.skip_val and iteration % val_per_iters == 0:         
                metadata = self.validation(table)   
                for image, mask, predicted_mask in zip(images, targets, outputs[0]):
                    image = F.to_pil_image(image)
                    mask = Image.fromarray(mask_to_color(mask.cpu().numpy()))
                    prediction_mask = Image.fromarray(mask_to_color(predicted_mask.argmax(0).cpu().numpy()))
                    table.add_data(*[wandb.Image(ImageOps.scale(image, min(1, 256/args.crop_size), Image.NEAREST))
                                     for image in (image, mask, prediction_mask)])
                self.model.train()
            else:
                metadata = {}
            
            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.model, self.args, is_best=False, metadata=metadata)


            wandb.log({'training_losses/total' : losses_reduced.item(),
                      **{'training_losses/'+label : loss.item() for label, loss in loss_dict.items()},
                      **metadata})

        wandb.log({'tech_debt':table})
        
        save_checkpoint(self.model, self.args, metadata={}, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(f"Total training time: {total_training_str} ({total_training_time/max_iters:.4f}s / it)")
        wandb.finish()
        
    def validation(self, table):
        is_best = False
        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)

            
            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            logger.info(f"Sample: {i+1:d}, Validation pixAcc: {pixAcc:.3f}, mIoU: {mIoU:.3f}")
                
        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        metadata = {'test_score': mIoU,
                    'test_metrics/pixelAcc': pixAcc,
                    'test_metrics/mIoU': mIoU}
        save_checkpoint(self.model, self.args, metadata, is_best)
        synchronize()
        return metadata


def save_checkpoint(model, args, metadata, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    base = f'{args.model}_{args.backbone}_{args.dataset}'
    filename = base + '.pth'
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    artifact = wandb.Artifact(base, type='weights')
    artifact_filename = time.strftime("/tmp/%y%m%dT%H%M%S_") + base + '.pth'
    shutil.copyfile(filename, artifact_filename)
    artifact.add_file(artifact_filename)
    for key, value in metadata.items():
        artifact.metadata[key] = value
    wandb.log_artifact(artifact)
    if is_best:
        best_filename = base + '_best_model.pth'
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)

if __name__ == '__main__':
    args = parse_args()

    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    args.lr = args.lr * num_gpus

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(),
                          filename=f'{args.model}_{args.backbone}_{args.dataset}_log.txt')
    logger.info(f"Using {num_gpus} GPUs")
    logger.info(args)
    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
