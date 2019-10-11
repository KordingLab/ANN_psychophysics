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

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--rotation', default=0, type=float,
                    help='Degrees of rotation on the input images')

parser.add_argument('--n-batches', default=1000, type=int,
                    help='Degrees of rotation on the input images')

parser.add_argument('--hamming', action='store_true',
                    help='Wether to hamming filter image')



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
        return transforms.functional.rotate(img, args.rotation)
   
    

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            rotate_img,
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.Grayscale(),
            transforms.ToTensor(),
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, pin_memory=True, sampler=None)


    fft_getter = FFT()

    # train for one epoch
    get_fft(train_loader, fft_getter,args)


def get_fft_simulated(fft_getter):


    im = torch.from_numpy(broadband_noise(size=224, 
                    orientation=40/180*np.pi, ))
    
    im = im.view(1,1,224,224)

    fft_im = fft_getter(im)

    return fft_im



def get_fft(train_loader, fft_getter,args):

    all_ffts = []
    
    hamm = torch.hamming_window(224, periodic=True, alpha=0.5, beta=0.5,)
    hamm_2d = torch.matmul(hamm[None].transpose(0,1), hamm[None]).cuda(args.gpu)
    
    for i, (images, _) in enumerate(train_loader):
        print(i)
        if args.hamming:
            images = images.cuda(args.gpu) * hamm_2d
        else:
            images = images.cuda(args.gpu)
        if i<1:
            for j in range(10):
                save_image(images[j], "unit_tests/hamming/{}.png".format(j))
                
        # get the ftt of each grayscale image
        mean_fft = fft_getter(images).cpu()

        all_ffts.append(mean_fft)
        np.save("ffts/mean_fft_unshifted_{}".format(i),mean_fft)
        if i>args.n_batches:
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

