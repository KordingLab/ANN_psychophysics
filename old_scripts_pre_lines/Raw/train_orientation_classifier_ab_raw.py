import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
from tqdm import tqdm
import torch
import torchvision
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
from torch.autograd import Variable


KERNEL_SIZE = 30
assert KERNEL_SIZE in (5,10,15,20,25,30)


def features_dataset(features_pickle_path, train=True):
    """
    Loads the precomputed features.

    Returns a pytorch Dataset
    """
    if train:
        start = 0
        stop = 48000
    else:
        start = 48000
        stop = 50000

    features = pd.read_hdf(features_pickle_path, start=start, stop=stop).values

    labels = torch.zeros(features.shape[0]).float()
    features = torch.from_numpy(features).float()

    return torch.utils.data.TensorDataset(features, labels)


LAYER_NUMBER = 24
assert LAYER_NUMBER in (24,31)

feature_num_dict = {24: 100352, 31 :512*7*7}
num_features = feature_num_dict[LAYER_NUMBER]



# features_data_train = features_dataset('fast_data/features/vgg_maxpool{}.h5'.format(LAYER_NUMBER), train = True)
features_data_valid = features_dataset('fast_data/features/vgg_maxpool{}.h5'.format(LAYER_NUMBER), train = False)


def orientations_generator(h5_path, batch_size, as_tensor=True, train=True):
    if train:
        start = 0
        stop = 48000
    else:
        start = 48000
        stop = 50000

    curr_index = start
    while 1:

        dataframe = pd.read_hdf(h5_path, start=curr_index,
                                stop=min([curr_index + batch_size, stop]))
        #         print("Indexes {} to {}".format(curr_index,curr_index+batch_size))
        curr_index += batch_size

        if curr_index >= stop:
            curr_index = start
            continue

        if as_tensor:
            out = torch.Tensor(dataframe.values).view(-1, 2, 224, 224)
        else:
            out = dataframe.values.reshape((-1, 2, 224, 224))
        yield out


def orientations_iterator(h5_path, batch_size, as_tensor=True, train=True):
    return iter(orientations_generator(h5_path, batch_size, as_tensor, train))


import torch.nn as nn


class color_predictor(torch.nn.Module):
    def __init__(self, num_ftrs):
        super(color_predictor, self).__init__()

        if num_ftrs == 512 * 7 * 7:
            # this means we're using the maxpool 31 layer
            self.layer = "31"
        elif num_ftrs == 512 * 14 * 14:
            # this means we're using the maxpool 24 layer
            self.layer = "24"
        else:
            raise NotImplementedError("Need to set up the decoder for this layer")

        if self.layer == "31":
            self.first_deconv = torch.nn.Sequential(

                torch.nn.ConvTranspose2d(512, 64, 6, 4, 1),
                nn.BatchNorm2d(64),
                nn.ReLU())

        elif self.layer == "24":
            self.first_deconv = torch.nn.Sequential(

                torch.nn.ConvTranspose2d(512, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU())

        self.deconv = torch.nn.Sequential(

            # 64 x 28 * 28
            torch.nn.ConvTranspose2d(64, 16, 4, 2, 1),
            #             nn.BatchNorm2d(16),
            nn.ReLU(),
            # 16 x 56 x 56
            torch.nn.ConvTranspose2d(16, 2, 6, 4, 1),
            # 2 x 224 x 224
        )

    def forward(self, x):
        if self.layer == "31":
            x = x.view(-1, 512, 7, 7)
        elif self.layer == "24":
            x = x.view(-1, 512, 14, 14)

        x = self.first_deconv(x)
        x = self.deconv(x)

        return x


model = color_predictor(num_features).cuda()

TEST_BATCH_SIZE = 64
BATCH_SIZE = 128
LOG_INTERVAL = 5
EPOCHS = 10

features_loader = torch.utils.data.DataLoader(features_data_train,
                                              batch_size=BATCH_SIZE, shuffle=False,
                                              num_workers=2, pin_memory=True, drop_last=True)

orientation_loader = orientations_iterator('data/features/orientations/{}/all_images.h5'.format(KERNEL_SIZE),
                                           BATCH_SIZE, train=True)

valid_features_loader = torch.utils.data.DataLoader(features_data_valid,
                                                    batch_size=TEST_BATCH_SIZE, shuffle=False,
                                                    num_workers=2, pin_memory=True, drop_last=True)

valid_orientation_loader = orientations_iterator('data/features/orientations/{}/all_images.h5'.format(KERNEL_SIZE),
                                                 TEST_BATCH_SIZE, train=False)

optimizer = torch.optim.Adam(model.parameters(), 1e-3,
                             weight_decay=1e-4)

criterion = torch.nn.MSELoss()


def train(epoch):
    test_accuracy = []
    train_accuracy = []
    model.train()
    train_loss = 0
    i = 0
    for batch_idx, ((feats, _), orients) in enumerate(zip(features_loader, orientation_loader)):

        data, target = feats.cuda(), orients.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output ** 3, target ** 3)
        loss.backward()

        train_loss += loss.data
        i += data.size()[0]

        optimizer.step()

        if (batch_idx > LOG_INTERVAL) and (batch_idx % LOG_INTERVAL == 0):
            train_loss /= i / BATCH_SIZE
            i = 0

            train_accuracy.append(train_loss)
            test_loss = test()
            test_accuracy.append(test_loss)

            print('Epoch: {} Train Loss: {:.6f} || Test Loss: {:.6f}  '.format(
                epoch + 1,
                train_loss, test_loss))
            train_loss = 0

    return train_accuracy, test_accuracy


def test():
    model.eval()
    test_loss = 0
    i = 0
    n_val = 2

    for batch_idx, ((feats, _), orients) in enumerate(zip(valid_features_loader, valid_orientation_loader)):
        data, target = feats.cuda(), orients.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.data
        i += 1
        if i > n_val:
            break

    test_loss /= i

    model.train()

    return test_loss


def adjust_opt(optimizer, epoch):
    if (epoch + 1) % 5 == 0:
        print('Adjusted')
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.


test_accuracy = []
train_acc = []
for epoch in range(EPOCHS):
    tr, te = train(epoch)
    test_accuracy += te
    train_acc += tr
    adjust_opt(optimizer, epoch)

plt.plot(np.array(test_accuracy))
plt.plot(np.array(train_acc))

plt.plot(np.array(test_accuracy))
plt.plot(np.array(train_acc))
# plt.ylim(0,200)

plt.savefig('sdfasd.png')

plt.plot(np.array(test_accuracy))
plt.plot(np.array(train_acc))
# plt.ylim(0,200)

torch.save(model.state_dict(), "/home/abenjamin/DNN_illusions/data/models/aritrained_kernel-10.pt")
print("Saved")


valdir = '/data2/imagenet/val'
# Load the images for visualizing
val_dataset_nonnormalized = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]))


class SubsetSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


from colorspacious import cspace_convert

from matplotlib.colors import ListedColormap


def plot_viewable(x, y):
    magnitude = torch.sqrt(x ** 2 + y ** 2)
    magnitude /= torch.max(magnitude)

    angle = torch.atan2(y, x)

    color_circle = np.ones((256, 3)) * 66
    color_circle[:, 1] = np.ones((256)) * 44

    color_circle[:, 2] = np.arange(0, 360, 360 / 256)

    color_circle_rgb = cspace_convert(color_circle, "JCh", "sRGB1")
    cm = ListedColormap(color_circle_rgb)

    a = plt.imshow(angle,
                   cmap=cm,
                   vmin=-np.pi,
                   vmax=np.pi)

    plt.imshow(1 - magnitude, vmin=0, vmax=1, alpha=.6)

    cbar = plt.colorbar(a, aspect=10, fraction=.07)
    cbar.ax.set_ylabel('Phase [pi]')
#     plt.show()


# get predicted outputs for some inputs

TEST_BATCH_SIZE = 64
BATCH_SIZE = 2
LOG_INTERVAL = 100
EPOCHS = 10

features_loader_2 = torch.utils.data.DataLoader(features_data_valid,
                                                batch_size=BATCH_SIZE, shuffle=False,
                                                num_workers=4, pin_memory=True, drop_last=True)

orientation_loader_10 = orientations_iterator('data/features/orientations/{}/all_images.h5'.format(10),
                                              BATCH_SIZE, train=False)
orientation_loader_30 = orientations_iterator('data/features/orientations/{}/all_images.h5'.format(30),
                                              BATCH_SIZE, train=False)

image_loader = torch.utils.data.DataLoader(val_dataset_nonnormalized,
                                           batch_size=BATCH_SIZE, shuffle=False,
                                           sampler=SubsetSampler(range(48000, 50000)),
                                           num_workers=1, pin_memory=True)

for batch_idx, ((feats, _), orients10, orients30, (images, _)) in enumerate(zip(features_loader_2,
                                                                                orientation_loader_10,
                                                                                orientation_loader_30,
                                                                                image_loader)):
    data1 = feats.cuda()
    target = orients30.cuda()
    data1 = Variable(data1)

    #         output = model(data1)

    for index in range(BATCH_SIZE):
        plt.figure(figsize=(10, 10))
        #             plt.subplot(221)
        #             x,y = output[index].detach().cpu()
        #             plot_viewable(x, y)
        #             plt.title("Model output")
        #             plt.axis("off")

        plt.subplot(222)
        x, y = target[index].detach().cpu()
        plot_viewable(x, y)
        plt.title("Loaded orientations: 30")
        plt.axis("off")

        plt.subplot(224)
        x, y = orients10[index].detach().cpu()
        plot_viewable(x, y)
        plt.title("Loaded orientations: 10")
        plt.axis("off")

        plt.subplot(223)
        image = images[index]
        image = np.squeeze(np.moveaxis(image.numpy(), 0, -1))
        plt.imshow(image)
        plt.title("Da pic")
        plt.axis("off")

    #             magnitude, angle = to_viewable_image(three_d_output)
    #             magnitude1, angle1 = to_viewable_image(three_d_orient)
    #             print("Output Image")
    #             plot_viewable(magnitude, angle)
    #             print("Original Orientation")
    #             plot_viewable(magnitude1, angle1)

    break
print("done")

# compare with the function above


def genGabor(sz, omega, theta, func=np.cos, K=np.pi):
    """Thanks to Prof. Dan Kersten's psy5046 course notes

    Inputs: sz - tuple, filter size
            omega - together with K, controls frequency and size
            theta - orientation of filter
            K - sinusoid only; strength of modulation
            """

    radius = (int(sz[0] / 2.0), int(sz[1] / 2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0] + 1), range(-radius[1], radius[1] + 1))

    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)

    gauss = omega ** 2 / (4 * np.pi * K ** 2) * np.exp(- omega ** 2 / (8 * K ** 2) * (4 * x1 ** 2 + y1 ** 2))
    sinusoid = func(omega * x1) * np.exp(K ** 2 / 2)
    gabor = gauss * sinusoid
    return gabor


def get_quadratures(kernel_size):
    filts = [(genGabor((kernel_size, kernel_size), 24 / kernel_size, angle, np.cos, 2),
              genGabor((kernel_size, kernel_size), 24 / kernel_size, angle, np.sin, 2))
             for angle in np.arange(0, np.pi, np.pi / 4)]

    return filts


from scipy.signal import convolve2d


def get_orientation_map_tensor(image, filts, rescale_angle=False, max_intensity=220):
    """Convolves 4 quadrature filters over an image and returns the image orientation.
    Converts to greyscale first.

    Inputs: image - (1,224,224) pytorch Tensor (greyscale image)
            filts -

    Outputs: (2,224,224) tensor, with first channel the x of the orientation vectors at each
                                pixel and the second channel the y's

    """
    # move to numpy
    image = np.squeeze(image.numpy())

    # convolve Gabors and get energy of each
    magnitudes = []
    for filt in filts:
        sin_conv = convolve2d(image, filt[1], mode='same')
        cos_conv = convolve2d(image, filt[0], mode='same')

        magnitudes.append(np.sqrt(sin_conv ** 2 + cos_conv ** 2))

    orientation_vec = np.array([magnitudes[0] - magnitudes[2],
                                magnitudes[1] - magnitudes[3]])

    return orientation_vec


valdir = '/data2/imagenet/val'

filts = get_quadratures(KERNEL_SIZE)

orientation_maps = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: get_orientation_map_tensor(x, filts))
    ]))

bs = 3
image_loader = torch.utils.data.DataLoader(val_dataset_nonnormalized,
                                           batch_size=bs, shuffle=False,
                                           num_workers=1, pin_memory=False)

new_orients = torch.utils.data.DataLoader(orientation_maps,
                                          batch_size=bs, shuffle=False,
                                          num_workers=1, pin_memory=False)

orientation_loader = orientations_iterator('fast_data/features/orientations/{}/all_images.h5'.format(KERNEL_SIZE),
                                           bs, train=True)

i_s = []
for ind, ((images, _), orient, (new_orient, _)) in enumerate(zip(image_loader, orientation_loader, new_orients)):

    for index in range(bs):
        plt.figure(figsize=(10, 30))
        plt.subplot(131)

        x, y = orient[index]
        plot_viewable(x, y)
        plt.title("Orientations")
        plt.axis("off")

        plt.subplot(132)
        x, y = new_orient[index]
        plot_viewable(x, y)
        plt.title("Recalculated orientations")
        plt.axis("off")

        plt.subplot(133)
        image = images[index]
        image = np.squeeze(np.moveaxis(image.numpy(), 0, -1))
        plt.imshow(image)
        plt.title("Da pic")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
    break


from tqdm import tqdm as tqdm

BATCH_SIZE = 64

for KERNEL_SIZE in [10, 20, 30]:

    filts = get_quadratures(KERNEL_SIZE)

    orientation_maps = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: get_orientation_map_tensor(x, filts))
        ]))

    orientation_loader = torch.utils.data.DataLoader(orientation_maps,
                                                     batch_size=BATCH_SIZE, shuffle=False,
                                                     num_workers=8, pin_memory=True, drop_last=False)

    all_maps = np.zeros((50000, 2 * 224 * 224))