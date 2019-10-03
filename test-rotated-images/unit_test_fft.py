import argparse
import os
import random
import shutil
import time
import warnings
import pickle
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import save_image
import torchvision
from fisher_calculators import get_fisher_orientations as get_fisher_now
from fisher_calculators import gen_sinusoid

import sys
sys.path.insert(1, '../single_patch_orientation')
from orientation_stim import broadband_noise

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')



best_acc1 = 0


def main():
    args = parser.parse_args()

    main_worker( args)


def main_worker(args):

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def rotate_img(img):
        return transforms.functional.rotate(img, 45)

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            rotate_img,
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.Grayscale(),
            transforms.ToTensor(),
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, pin_memory=True, sampler=None)


    fft_getter = FFT()

    # test on simulated data
#    test_im = get_fft_simulated(fft_getter).numpy()
#    print("Output size of fft",test_im.shape)

#    np.save("test_fft_sin",np.fft.fftshift(test_im))

    # train for one epoch
    get_fft(train_loader, fft_getter)


def get_fft_simulated(fft_getter):


    im = torch.from_numpy(broadband_noise(size=224, 
                    orientation=40/180*np.pi, ))
    
    im = im.view(1,1,224,224)

    fft_im = fft_getter(im)

    return fft_im



def get_fft(train_loader, fft_getter):

    all_ffts = []
    for i, (images, _) in enumerate(train_loader):
        print(i)
        # get the ftt of each grayscale image
        mean_fft = fft_getter(images.cuda()).cpu()

        all_ffts.append(mean_fft)
        np.save("ffts/mean_fft_unshifted_{}".format(i),mean_fft)
        if i>100:
            break
    all_mean_fft = torch.mean(torch.stack(all_ffts),dim=0)
    np.save("ffts/all_mean_fft_unshifted",all_mean_fft)
        


class FFT(torch.nn.Module):
    """When called this 'network' takes the spatial FFT of a batch of inputs.

    Returns a (224,224) torch tensor.
    """

    def __init__(self):
        super(FFT, self).__init__()

    def forward(self, x):
        # should be shape [BS, 1, 224, 224]
#        assert list(x.size()[1:]) == [1, 224, 224]

        ffted_inputs = torch.rfft(x, signal_ndim=2, onesided=False)
        # now shape [BS, 1, 224, 224, 2]

        # get the magnitude of this complex number
        fft_magnitude = torch.norm(ffted_inputs, dim = 4)

        # average over this batch
        mean_fft = torch.mean(fft_magnitude,dim=0)
        # now shape [1,224,113]

        return mean_fft[0]



if __name__ == '__main__':
    main()

