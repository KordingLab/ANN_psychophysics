
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models

import argparse
import pickle

from scipy.ndimage import rotate

from decoder_upsample_nonlinear import VGG_chopped

'''
Calculates the Fisher information of a certain layer of the VGG network w/r/t the angle of intersecting lines.

Uses a finite-difference method:
- we generate an image of two intersecting lines...
- get the network responses...
- generate another image of two lines, cocentric with the first but +/- a small number of degrees
- get the responses...
- get the difference
'''


def generate_intersecting_rgb(centerloc, angle, relative_angle, linewidth = 1):
    """

    :param centerloc: The location of the intersection, in pixels (0,224)
    :param angle: The angle of the midpoint of the intersection
    :param relative_angle: The angle between the two intersecting lines
    :param linewidth: How wide the lines are
    :return: An RGB numpy array of shape (224,224,3) corresponding to the image
    """

    starter = np.ones((224 * 4, 224 * 4, 3))

    x = np.cos(angle)
    y = np.sin(angle)

    # reverse polarity
    centerloc = np.array([224,224])-centerloc

    # start with horizontal bar at center
    starter[(112 * 4 - linewidth):(112 * 4 + linewidth)] = 0

    # Rotate it
    angle1 = angle+relative_angle/2
    im = rotate(starter, angle1 * 180 / np.pi, axes=(0, 1), reshape=False)
    im = im[224 + centerloc[0]:224 * 2 + centerloc[0], 224 + centerloc[1]:224 * 2 + centerloc[1]]

    # Do the other
    angle2 = angle - relative_angle / 2
    im2 = rotate(starter, angle2 * 180 / np.pi, axes=(0, 1), reshape=False)
    im2 = im2[224 + centerloc[0]:224 * 2 + centerloc[0], 224 + centerloc[1]:224 * 2 + centerloc[1]]

    # combine. Note by taking the average the lines are overlaid with .50 alpha.
    im = (im + im2) / 2

    return im


def numpy_to_torch(rgb_image):
    """
    Prepares an image for passing through a pytorch network.
    :param rgb_image: Numpy tensor, shape (x,y,3)
    :return: Pytorch tensor, shape (3,x,y), with channels switched to BGR.

    >>> numpy_to_torch(np.ones((224,224,3))).size()
    torch.Size([3, 224, 224])
    """

    #rgb to bgr
    tens = torch.from_numpy(rgb_image[:,:,[2,1,0]])
    return tens.permute(2,0,1)

def get_vgg_response(image, layer, network, centerloc, image_frac = .2):
    """
    Gets the response of the network of the VGG at a certain layer to a single image
    NOTE: we only return the activations corresponding to the center of the image.

    Assumes the network is on the GPU already.

    :param image: An RGB numpy image
    :param layer: which layer? 4,9,16,23,30
    :param network: instance of the network on the GPU to run images through
    :param centerloc: location of the intersection, in image space
    :param image_frac: We only take some amount of the network activations. What fraction to take?
                            (measured in the fraction of the grid on either side around the intersection)
    :return: Pytorch tensor of the activations
    """
    torch_image = numpy_to_torch(image).float().cuda()
    # indexing by None adds a new axis in the beginning
    out = network(torch_image[None])

    out = _crop_activations(out,centerloc,image_frac)

    return out

def _crop_activations(out,centerloc,image_frac):
    """
    Helper for get_vgg_response. Takes a tensor of activations, chops it up so we only get the activations
    nearby to the line intersection
    :param out: the activations
    :param centerloc: Location of the line intersection
    :param image_frac: Same as in get-vgg-response
    :return: The cropped activations, flattened

    """
    grid_size = out.size()[-1]

    # put centerloc in activation space
    centerloc = np.array(centerloc) // (224/grid_size)
    left = int(centerloc[0] - grid_size * image_frac)
    right = int(centerloc[0] + grid_size * image_frac)
    top = int(centerloc[1] - grid_size * image_frac)
    bottom = int(centerloc[1] + grid_size * image_frac)

    out = out[0, :, left:right, top:bottom].contiguous().view(-1)
    assert len(out) > 0, "image_frac too small: selected just one grid"
    return out


def get_derivative(delta, plus_resp, minus_resp):
    """
    Calculates the finite-difference derivative of the network activation w/r/t the relative angle between two lines
    :param delta: The perturbation
    :param minus_resp: The activations at the negative perturbation
    :param plus_resp: The activations at the positive perturbation

    :return: The derivative of the activations of the network at specified layer

    >>> f = lambda x: x**2
    >>> d = get_derivative(1e-3, f(2+1e-3), f(2-1e-3))
    >>> assert np.isclose(d,4)
    >>> d = get_derivative(1e-3, f(5+1e-3), f(5-1e-3))
    >>> assert np.isclose(d,10)
    """

    deriv = (plus_resp - minus_resp)/ (2 * delta)
    return deriv

def get_fisher(df_dtheta):
    """
    Compute fisher information under Gaussian noise assumption

    :param df_dtheta: Derivative of a function f w/r/t some parameter theta
    :return: The Fisher information (1-dimensional)
    """

    fisher =  torch.dot(df_dtheta, df_dtheta).item()
    return fisher


def plot_fisher(angles,fishers,savedir):
    """
    Plots the fishers
    :param angles:
    :param fishers:
    :return:
    """
    plt.figure(figsize = (5,5))
    plt.plot(angles,fishers,"-o")
    plt.xlabel("Angle",fontsize = 15)
    plt.ylabel("Fisher Info", fontsize = 15)
    plt.tight_layout()
    plt.savefig(savedir+"/fisher_plot.png")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images",
                        help="How many images to use in the finite difference calculation",
                        type=int,
                        default = 1000)
    parser.add_argument("layer", help="which layer of VGG the model was trained to decode from",
                        type=int)
    parser.add_argument("--random-center",
                        help = "Randomly sampling the center point of the intersection",
                        action = 'store_true')
    parser.add_argument("--delta",
                        help = "Size, in radians, of the difference on either side of chosen distance",
                        type = float,
                        default = 1e-3)
    parser.add_argument("--savename",
                help = "Relative path to a folder in which to save: - a pickle containing the Fisher; - a plot of it. ",
                        type = str,
                        default = './')

    args = parser.parse_args()


    maxpool_indices = [4, 9, 16, 23, 30]
    assert args.layer in maxpool_indices

    #build network
    vgg_chopped = VGG_chopped(args.layer).cuda()

    angles = np.linspace(-np.pi,np.pi,args.n_images)
    fishers = []
    for relative_angle in angles:
        # decide on the intersecting location
        centerloc = (112,112)
        if args.random_center:
            centerloc = np.random.randint(0, 224, 2)

        # randomly sample a main angle
        main_angle = 2*np.pi*np.random.uniform()-np.pi

        # get the image
        plus_image = generate_intersecting_rgb(centerloc, main_angle, relative_angle+args.delta)
        minus_image = generate_intersecting_rgb(centerloc, main_angle, relative_angle-args.delta)

        # get the response
        plus_resp = get_vgg_response(plus_image, args.layer, vgg_chopped, centerloc)
        minus_resp = get_vgg_response(minus_image, args.layer ,vgg_chopped, centerloc)

        # get the Fisher. Now working in pytorch
        df_dtheta = get_derivative(args.delta, plus_resp, minus_resp)
        fishers.append(get_fisher(df_dtheta))

    # save this data

    pickle.dump((angles,fishers), open(args.savename+"/fisher.pickle",'wb'))

    plot_fisher(angles,fishers, args.savename)


