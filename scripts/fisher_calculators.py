
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models, transforms

import argparse
import pickle


############################ Generators from Linqi's code ##########################################

def gen_sinusoid(sz, A, omega, rho):
    radius = int(sz / 2.0)
    [x, y] = torch.meshgrid([torch.tensor(range(-radius, radius)),
                             torch.tensor(range(-radius, radius))])
    x = x.float()
    y = y.float()
    stimuli = A * torch.cos(0.35 * omega[0] * x + 0.35 * omega[1] * y + rho)
    return stimuli


def gen_sinusoid_aperture(ratio, sz, A, omega, rho, polarity):
    sin_stimuli = gen_sinusoid(sz, A, omega, rho)
    radius = int(sz / 2.0)
    [x, y] = torch.meshgrid([torch.tensor(range(-radius, radius)),
                             torch.tensor(range(-radius, radius))])
    aperture = torch.empty(sin_stimuli.size(), dtype=torch.float)

    aperture_radius = float(radius) * ratio
    aperture[x ** 2 + y ** 2 >= aperture_radius ** 2] = 1 - polarity
    aperture[x ** 2 + y ** 2 < aperture_radius ** 2] = polarity

    return sin_stimuli * aperture

def gen_hamming_aperature_hamming():
    raise NotImplementedError("Lingqi!")


def center_surround(ratio, sz, theta_center, theta_surround, A, rho):
    center = gen_sinusoid_aperture(ratio, sz, A, [torch.cos(theta_center), torch.sin(theta_center)], rho, 1)
    surround = gen_sinusoid_aperture(ratio, sz, A, [torch.cos(theta_surround), torch.sin(theta_surround)], rho, 0)
    return center + surround


def sinsoid_noise(ratio, sz, A, omega, rho):
    radius = int(sz / 2.0)
    sin_aperture = gen_sinusoid_aperture(ratio, sz, A, omega, rho, 1)

    nrm_dist = normal.Normal(0.0, 0.12)
    noise_patch = nrm_dist.sample(sin_aperture.size())

    [x, y] = torch.meshgrid([torch.tensor(range(-radius, radius)),
                             torch.tensor(range(-radius, radius))])
    aperture = torch.empty(sin_aperture.size(), dtype=torch.float)

    aperture_radius = float(radius) * ratio
    aperture[x ** 2 + y ** 2 >= aperture_radius ** 2] = 1
    aperture[x ** 2 + y ** 2 < aperture_radius ** 2] = 0

    return noise_patch * aperture + sin_aperture


def rgb_sinusoid(theta):
    output = torch.zeros(1, 3, 224, 224)
    sin_stim = gen_sinusoid(224, A=1, omega=[torch.cos(theta), torch.sin(theta)], rho=0)
    for idx in range(3):
        output[0, idx, :, :] = sin_stim

    return output


def rgb_sine_aperture(theta):
    output = torch.zeros(1, 3, 224, 224)
    sin_stim = gen_sinusoid_aperture(0.85, 224, A=1, omega=[torch.cos(theta), torch.sin(theta)], rho=0, polarity=1)
#     show_stimulus(sin_stim)
    for idx in range(3):
        output[0, idx, :, :] = sin_stim

    return output


def rgb_sine_noise(theta):
    output = torch.zeros(1, 3, 224, 224)
    sin_stim = sinsoid_noise(0.75, 224, A=1, omega=[torch.cos(theta), torch.sin(theta)], rho=0)
    for idx in range(3):
        output[0, idx, :, :] = sin_stim

    return output


def rgb_center_surround(theta_center, theta_surround):
    output = torch.zeros(1, 3, 224, 224)
    stimulus = center_surround(0.75, 224, theta_center, theta_surround, A=1, rho=0)
    for idx in range(3):
        output[0, idx, :, :] = stimulus
    return output


def show_stimulus(I):
    plt.figure()
    plt.axis('off')
    plt.imshow(I.detach().numpy(), cmap=plt.gray())
    plt.show()


##################################################################################################

def get_fisher_hues(model, layer, n_hues=120,  delta = 1e-2, generator = None, hues=None):
    """ Takes a full model (unchopped) along with a layer specification, and returns the fisher information with respect to hue of that layer

    Also allows choosing the finite-difference delta

    :param n_images: number of times to call generator and do the finite difference calculation. 
                     Averages over all derivatives to give a single Fisher.
    :param generator: if not None, a callable image generator taking "angle" as an input.
                      Should return a (3,224,224) Tensor showing a grating with orientation == angle.
                      If None, uses rbg_sine_aperature"""


    if hues==None:
        angles = np.linspace(0, 360, n_hues)
    else:
        angles=hues
                                                                

    fishers_at_angle = []
    for angle in angles:
        #print("\n angle",angle)
        
        all_phases_plus = generator(angle +delta).cuda()
        all_phases_minus = generator(angle -delta).cuda()

        # get the response
        plus_resp = get_response(all_phases_plus, model, layer)

        size = plus_resp.size()

        minus_resp = get_response(all_phases_minus, model, layer)

        # get the derivative. Now working in pytorch
        df_dtheta = get_derivative(delta, plus_resp, minus_resp)
        df_dtheta = df_dtheta.view(1,-1)
        

        # average down
        fisher = get_fisher(df_dtheta)
        fishers_at_angle.append(fisher)
    # print("fisher",fisher)
    return fishers_at_angle


def get_fisher_orientations(model, layer, n_angles=120, n_images=1, delta = 1e-2, generator = None):
    """ Takes a full model (unchopped) along with a layer specification, and returns the fisher information
    with respect to orientation of that layer (averaged over phase of sine grating).

    Also allows choosing the finite-difference delta

    :param n_images: number of times to call generator and do the finite difference calculation. 
                     Averages over all derivatives to give a single Fisher.
    :param generator: if not None, a callable image generator taking "angle" as an input.
                      Should return a (3,224,224) Tensor showing a grating with orientation == angle.
                      If None, uses rbg_sine_aperature"""


    phases = np.linspace(0, np.pi, n_images)
    angles = np.linspace(0, np.pi, n_angles)


    # create negative mask. This is a circle centered in the middle of radius 100 pixels
    unit_circle = np.zeros((224, 224)).astype(bool)
    for i in range(224):
        for j in range(224):
            if (i - 112) ** 2 + (j - 112) ** 2 >= 50 ** 2:
                unit_circle[i, j] = True
                
    if generator is None:
        generator = lambda angle: rgb_sine_aperture(torch.tensor( angle))
                                                                 

    fishers_at_angle = []
    for angle in angles:
        #         print("\n angle",angle)
        """I'll put all phases in one giant tensor for faster torching"""
        all_phases_plus = torch.zeros(n_images ,3 ,224 ,224).cuda()
        all_phases_minus = torch.zeros(n_images ,3 ,224 ,224).cuda()

        for i ,phase in enumerate(phases):


            all_phases_plus[i] = generator(angle +delta, spatial_phase=phase)
            all_phases_minus[i] = generator(angle -delta,spatial_phase=phase)

        # get the response
        plus_resp = get_response(all_phases_plus, model, layer)

        size = plus_resp.size()

        minus_resp = get_response(all_phases_minus, model, layer)

        # get the derivative. Now working in pytorch
        df_dtheta = get_derivative(delta, plus_resp, minus_resp)
        # reshape to be in terms of examples
        df_dtheta = df_dtheta.view(n_images ,-1)

        # average down
        fisher = get_fisher(df_dtheta)
        fishers_at_angle.append(fisher)
    # print("fisher",fisher)
#     print("response size", size)
    return fishers_at_angle

def get_derivative(delta, plus_resp, minus_resp):
    """
    Calculates the finite-difference derivative of the network activation w/r/t the relative angle between two lines
    :param delta: The perturbation
    :param minus_resp: The activations at the negative perturbation
    :param plus_resp: The activations at the positive perturbation

    :return: The derivative of the activations of the network at specified layer. FLATTENED

    >>> f = lambda x: x**2
    >>> d = get_derivative(1e-3, f(2+1e-3), f(2-1e-3))
    >>> assert np.isclose(d,4)
    >>> d = get_derivative(1e-3, f(5+1e-3), f(5-1e-3))
    >>> assert np.isclose(d,10)
    """

    deriv = (plus_resp - minus_resp )/ (2 * delta)
    return deriv


def get_fisher(df_dtheta):
    """
    Compute fisher information under Gaussian noise assumption. 

    Averages over the 0th dimension. (Assumes they're specific examples)

    :param df_dtheta: Derivative of a function f w/r/t some parameter theta
    :return: The Fisher information (1-dimensional)
    """

    fishers = 0
    for d in df_dtheta:
        fishers += torch.dot(d ,d)
    fisher = fishers/ len(df_dtheta)

    return fisher


def numpy_to_torch(rgb_image, cuda=True):
    """
    Prepares an image for passing through a pytorch network.
    :param rgb_image: Numpy tensor, shape (x,y,3)
    :return: Pytorch tensor, shape (3,x,y), with channels switched to BGR.

    >>> numpy_to_torch(np.ones((224,224,3))).size()
    torch.Size([3, 224, 224])
    """

    # rgb to bgr
    tens = torch.from_numpy(rgb_image[:, :, [2, 1, 0]])
    if cuda:
        r = tens.permute(2, 0, 1).float().cuda()
    else:
        r = tens.permute(2, 0, 1).float()
    return r


def get_response(torch_image, model, layer):
    """
    Gets the response of the network of the VGG at a certain layer to a single image
    NOTE: we only return the activations corresponding to the center of the image.

    Assumes the network is on the GPU already.



    :param image: An torch image
    :param layer: which layer? 4,9,16,23,30
    :param hooked_model: A model with a 
    :return: Pytorch tensor of the activations
    """

    # preprocess image
    mean = torch.Tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
    std = torch.Tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    torch_image = (torch_image - mean) / std

    # indexing by None adds a new axis in the beginning
    if len(torch_image.size()) == 3:
        torch_image = torch_image[None]

        # register the hook
    outputs_at_layer = []

    def hook(module, input, output):
        outputs_at_layer.append(output.detach())

    # vgg/alexnet or resnet?
    if len(list(model.children())) > 4:
        handle = list(model.children())[layer].register_forward_hook(hook)
    else:
        handle = list(model.children())[0][layer].register_forward_hook(hook)

    _ = model(torch_image)

    # clean up
    handle.remove()
    r = outputs_at_layer[0]
    del outputs_at_layer
    return r