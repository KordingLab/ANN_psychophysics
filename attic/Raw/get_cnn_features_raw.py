

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
from tqdm import tqdm
import torch
import torchvision
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import h5py
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F


#a kernel

valdir = '/data2/imagenet/val'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])



val_dataset = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.CenterCrop(224),
#         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

val_dataset_nonnormalized = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]))

#a kernel

val_loader = torch.utils.data.DataLoader(val_dataset,
    batch_size=256, shuffle=False,
    num_workers=1, pin_memory=True)

#a kernel

## also add Gaussian noise to the images to make it harder?

def add_noise(tensor):
    noise = torch.randn(tensor.size())
    noise_level = .5
    tensor = tensor + noise_level * noise
    # clip to be within 0-1 range so the values are still colors
    tensor[tensor > 1.] = 1.
    tensor[tensor < 0.] = 0.

    return tensor


val_dataset_noised = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(add_noise),
    ]))

val_dataset_noised_normalized = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(add_noise),
        normalize
    ]))


#a kernel

from colorspacious import cspace_convert


def get_avg_color(image):
    """Takes an image, converts it to CIECAM02 UCS color space,
    grabs the center pixels, averages the color. We move to UCS
    because it's a perceptually uniform space, in which we can average.
    Then we move to CIECAM02 JCh space to return the hue angle.

    Inputs:
    image:  non-normalized (i.e. RGB 0-1 floats) numpy array of shape(224,224,3)"""

    perceptually_uniform = cspace_convert(image, "sRGB1", "CAM02-UCS")
    avg_center = np.mean(np.mean(perceptually_uniform, axis=0), axis=0)

    avg_center_JCh = cspace_convert(avg_center, "CAM02-UCS", "JCh")

    return avg_center_JCh


def image_w_center_box(image, sq_pixels):
    _min_, _max_ = 224 // 2 - sq_pixels // 2, 224 // 2 + sq_pixels // 2
    image[_min_, _min_:_max_] = np.max(image)
    image[_max_:_max_ + 2, _min_:_max_] = np.max(image)
    image[_min_:_max_, _max_:_max_ + 2] = np.max(image)
    image[_min_:_max_, _min_] = np.max(image)

    return image

#a kernel

def image_w_box(image_, sq_pixels, left, top):
    image = image_.copy()

    right, bottom = left + sq_pixels, top + sq_pixels
    image[left:left + 2, top:bottom] = np.max(image)
    image[right:right + 2, top:bottom] = np.max(image)
    image[left:right, bottom:bottom + 2] = np.max(image)
    image[left:right, top:top + 2] = np.max(image)

    return image


#a kernel
i = 0
for image, label in val_dataset:
    sq_pixels = 20
    left, top = 45, 106

    image = np.squeeze((np.moveaxis(image.numpy(), 0, -1)))
    plt.subplot(131)
    fig = plt.imshow(image_w_box(image, sq_pixels, left, top))
    plt.axis('off')

    plt.subplot(132)
    right, bottom = left + sq_pixels, top + sq_pixels
    image_box = image[left:right, top:bottom, :]
    fig = plt.imshow(image_box)
    plt.axis('off')

    plt.subplot(133)
    color = get_avg_color(image_box)

    test_image = np.ones((20, 20, 3)) * color
    converted = cspace_convert(test_image, "JCh", "sRGB1", )
    plt.imshow(converted)
    plt.axis('off')
    plt.show()
    i += 1
    if i > 5:
        break

#a kernel

def save_colors_in_box(sq_pixels, left, top):
    ## run for all validation images and save to file

    val_loader = torch.utils.data.DataLoader(val_dataset_nonnormalized,
                                             batch_size=1, shuffle=False,
                                             num_workers=1, pin_memory=True)

    center_colors = pd.DataFrame(columns=['J', 'C', 'h'])

    for i, (image, label) in tqdm(enumerate(val_loader)):
        # get first image in batch (of 1 anyways)
        image = image[0]
        image = np.squeeze(np.moveaxis(image.numpy(), 0, -1))

        # define box to grab color from

        right, bottom = left + sq_pixels, top + sq_pixels
        image_box = image[left:right, top:bottom, :]

        color = get_avg_color(image_box)

        name = val_dataset.imgs[i][0]
        center_colors.loc[name] = color

    center_colors.to_pickle('data/features/center_colors_{}_sq_left{}_top{}.pickle'.format(
        sq_pixels, left, top))

    return center_colors


#a kernel

# For the adelson: center, [115,60], [45, 106]
# sq_pixels = 20
# left, top = 115,60#45, 106

# # For the rubix cube
# sq_pixels = 14
# left, top = 46,121

# again for the cube but smaller
sq_pixels = 8
left, top = 142,98
center_colors = save_colors_in_box(sq_pixels, left, top)

left, top = 48,123
center_colors = save_colors_in_box(sq_pixels, left, top)



#a kernel

import torchvision.models as models
# resnet18 = models.resnet18(pretrained=True)
# alexnet = models.alexnet(pretrained=True)
# squeezenet = models.squeezenet1_0(pretrained=True)
model = models.vgg16(pretrained=True)
# densenet = models.densenet161(pretrained=True)
# inception = models.inception_v3(pretrained=True)


#a kernel

class VGG_l6(torch.nn.Module):

    def __init__(self, model, layer):
        super(VGG_l6, self).__init__()
        self.features = model.features
        self.classifier = list(model.classifier.children())[0]
        self.layer = layer

        self.maxpool_indices = [0, 5, 10, 17, 24, 31]

    def forward(self, x):
        for i in range(len(self.maxpool_indices) - 1):
            x = self.features[self.maxpool_indices[i]:self.maxpool_indices[i + 1]](x)
            if i == self.layer:
                return x
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# a kernel

bs = 64

features = ['vgg_maxpool5',
            'vgg_maxpool10',
            'vgg_maxpool17',
            'vgg_maxpool24',
            'vgg_maxpool31',
            'vgg_fc1']

# list of hdf stores
all_features = list()

# I've set this up so that it could save all the features each run. However for 50000 images this is too large
# for memory. Therefore I run a loop, and save a feature a iteration. Note that this requires running the network
# on the same image multiple times
for feature_number in range(len(features)):
    if feature_number < 5:
        continue
    print(features[feature_number])

    vgg_features = VGG_l6(model, feature_number).cuda()

    val_loader = torch.utils.data.DataLoader(val_dataset_noised_normalized,
                                             batch_size=bs, shuffle=False,
                                             num_workers=4, pin_memory=True)

    # switch to evaluate mode
    vgg_features.eval()

    for i, (input_var, _) in tqdm(enumerate(val_loader)):
        with torch.no_grad():
            input_var = torch.autograd.Variable(input_var).cuda()

        # compute features, one per layer
        feature_list = vgg_features(input_var)

        # add this particular feature to that list
        all_features.append(feature_list.detach().cpu().numpy())

    #         if (i%(len(val_loader)//10) == 0 ) and i>0:
    all_features = pd.DataFrame(np.vstack(all_features).reshape(50000, -1))
    all_features.to_hdf('data/features/{}_noised.h5'.format(features[feature_number]),
                        key='{}'.format(features[feature_number]), mode='w')
    del all_features
    all_features = list()




