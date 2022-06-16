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

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--n-images', default=100000000, type=int,
                    help='number of images to get over.')

parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

from colorspacious import cspace_converter

start_to_end_fn = cspace_converter('sRGB1', 'CIELCh')


def to_JCh_img(pil_imgs):
    """Takes a list of 5 pixels"""
    np_im = np.array([np.array(p) for p in pil_imgs]).squeeze()
    JCh_img = start_to_end_fn(np_im)

    return torch.from_numpy(JCh_img)


def main():
    args = parser.parse_args()

    main_worker(args)
    
def _rgb2hsv(img):
    r, g, b = img.unbind(dim=-3)

    # Implementation is based on https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/
    # src/libImaging/Convert.c#L330
    maxc = torch.max(img, dim=-3).values
    minc = torch.min(img, dim=-3).values

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occuring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    cr = maxc - minc
    # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = (hr + hg + hb)
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return torch.stack((h, s, maxc), dim=-3)

def rotate_hue(img):
    return transforms.functional.adjust_hue(img, .25)

def main_worker(args):
    # Data loading code
    traindir = os.path.join(args.data, 'train')

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            rotate_hue,
            transforms.CenterCrop(224),
            transforms.ToTensor()
            #transforms.FiveCrop(1), # just pick 5 pixels
            #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    hist = get_hue_histogram(train_loader, args)

    np.save("hue_hist_hsv/hist_all_pix_rot", hist)

from tqdm import tqdm as tqdm
def get_hue_histogram(train_loader, args):
    n_bins = 361
    histograms = torch.zeros(n_bins, dtype=torch.float)
    for i, (img, _) in tqdm(enumerate(train_loader)):
        if img.dtype == torch.uint8:
            img = img.to(dtype=torch.float32) / 255.0
        pixels = _rgb2hsv(img)
        h, s, v = pixels.unbind(dim=-3)
        h = h*360
        h[h > 360] -= 360
        h[s < .1] = -999
        h[v < .5] = -999
        #h[v > .9] = -999
        binned = torch.histc(h, bins=n_bins, min=-1, max=360).to(torch.float)
        histograms.add_(binned)

        if i > (args.n_images // args.batch_size):
            break
    return histograms


if __name__ == '__main__':
    main()

