import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as col

%matplotlib inline
import seaborn as sns
from tqdm import tqdm
import pickle

from scipy.signal import convolve2d

from colorspacious import cspace_convert


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

filts = get_quadratures(50)
for i in range(4):
    plt.subplot(121)
    plt.imshow(filts[i][1], vmin = -.1, vmax = .1)
    plt.subplot(122)
    plt.imshow(filts[i][0], vmin = -.1, vmax = .1)
    plt.show()

color_circle = np.ones((256, 3)) * 66
color_circle[:, 1] = np.ones((256)) * 44

color_circle[:, 2] = np.arange(0, 360, 360 / 256)
color_circle_rgb = cspace_convert(color_circle, "JCh", "sRGB1")

cm = col.ListedColormap(color_circle_rgb)
fig = plt.figure()
ax = plt.gca()

##### generate data grid
N = 256
x = np.linspace(-2, 2, N)
y = np.linspace(-2, 2, N)
z = np.zeros((len(y), len(x)))  # make cartesian grid
for ii in range(len(y)):
    z[ii] = np.arctan2(y[ii], x)  # simple angular function

pmesh = ax.pcolormesh(x, y, z / np.pi,
                      cmap=cm, vmin=-1, vmax=1)
plt.axis([x.min(), x.max(), y.min(), y.max()])
cbar = fig.colorbar(pmesh)
cbar.ax.set_ylabel('Phase [pi]')



def genSinusoid(sz, A, omega, rho):
    # Generate Sinusoid grating
    # sz: size of generated image (width, height)
    radius = (int(sz[0]/2.0), int(sz[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1)) # a BUG is fixed in this line

    stimuli = A * np.cos(omega[0] * x  + omega[1] * y + rho)
    return stimuli
theta = np.pi*0
omega = 1

plt.imshow(filts[0][0])

filts = get_quadratures(25)
theta = 90
sinusoid = genSinusoid((100, 100), 1, (omega * np.sin(theta), omega * np.cos(theta)), 0)

filt = filts[0]
sin_conv = convolve2d(sinusoid, filt[0], mode='same')

plt.imshow(sin_conv)

filts = get_quadratures(5)
for theta in np.arange(0, np.pi, np.pi / 8):
    sinusoid = genSinusoid((100, 100), 1, (omega * np.sin(theta), omega * np.cos(theta)), 0)
    magnitudes = []

    im = sinusoid
    for filt in filts:
        sin_conv = convolve2d(im, filt[1], mode='same')
        cos_conv = convolve2d(im, filt[0], mode='same')

        magnitudes.append(np.sqrt(sin_conv ** 2 + cos_conv ** 2))
    plt.subplot(121)
    plt.imshow(sinusoid)
    plt.subplot(122)

    orientation_vec = [magnitudes[0] - magnitudes[2],
                       magnitudes[1] - magnitudes[3]]
    a = plt.imshow(np.arctan2(magnitudes[0] - magnitudes[2], magnitudes[1] - magnitudes[3]),
                   cmap=cm, vmin=-np.pi, vmax=np.pi)

    plt.imshow(1 - np.sqrt(orientation_vec[0] ** 2 +
                           orientation_vec[1] ** 2) / \
               np.max(np.sqrt(orientation_vec[0] ** 2 +
                              orientation_vec[1] ** 2)), vmin=0, vmax=1, alpha=.3)

    cbar = plt.colorbar(a)
    cbar.ax.set_ylabel('Phase [pi]')
    plt.show()


from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"


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

## run for all validation images and save to file

val_loader = torch.utils.data.DataLoader(val_dataset_nonnormalized,
                                         batch_size=1, shuffle=False,
                                         num_workers=1, pin_memory=True)

#     center_colors = pd.DataFrame(columns=['J','C','h'])
filts = get_quadratures(24)

for i, (image, label) in tqdm(enumerate(val_loader)):
    # get first image in batch (of 1 anyways)
    image = image[0]
    image = np.squeeze(np.moveaxis(image.numpy(), 0, -1))

    # define box to grab color from
    #     convolve2d(image, filts[0][0],mode='valid')

    plt.subplot(121);
    plt.imshow(image);
    plt.title('Response of sin gabor(simple cell)')
    plt.show()
    if i > 1: break



perceptually_uniform = cspace_convert(image, "sRGB1", "JCh")
lightness = -perceptually_uniform[:,:,0]


magnitudes = []
filts = get_quadratures(30)
for filt in filts:
    sin_conv = convolve2d(lightness, filt[1],mode='same')
    cos_conv = convolve2d(lightness, filt[0],mode='same')

    magnitudes.append(np.sqrt(sin_conv**2+cos_conv**2))
plt.subplot(121)
plt.imshow(lightness)
plt.subplot(122)

orientation_vec = [magnitudes[0]-magnitudes[2],
                   magnitudes[1]-magnitudes[3]]
a =plt.imshow(np.arctan2(magnitudes[0]-magnitudes[2], magnitudes[1]-magnitudes[3]),
          cmap= cm, vmin = -np.pi, vmax = np.pi)

plt.imshow(1-np.sqrt(orientation_vec[0]**2+
                                 orientation_vec[1]**2)/\
                        np.max(np.sqrt(orientation_vec[0]**2+
                                 orientation_vec[1]**2)),vmin=0,vmax=1, alpha=.3)
cbar = plt.colorbar(a, aspect = 10, fraction = .07)
cbar.ax.set_ylabel('Phase [pi]')
plt.show()


from tqdm import tqdm as tqdm

from scipy.signal import convolve2d
from colorspacious import cspace_convert


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


BATCH_SIZE = 4

for KERNEL_SIZE in [30]:

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
                                                     num_workers=4, pin_memory=True, drop_last=False)

    all_maps = np.zeros((2, 2 * 224 * 224))
    for i, (maps, _) in enumerate(tqdm(orientation_loader)):
        for x, y in maps:
            plot_viewable(x, y)
            plt.show()
        raise
        batch = maps.detach().numpy().reshape(-1, 2 * 224 * 224)

        all_maps[BATCH_SIZE * i:BATCH_SIZE * (i + 1)] = batch

    all_maps = pd.DataFrame(all_maps)
    all_maps.to_hdf('fast_data/features/orientations/{}/all_images.h5'.format(KERNEL_SIZE),
                    key='kernel_{}'.format(KERNEL_SIZE), mode='w')
    del all_maps
#     all_maps = np.zeros((50000,2*224*224))


import gc
gc.collect()


def get_orientation_map(image, filts):
    """Convolves 4 quadrature filters over an image and returns the image orientation.
    Converts to greyscale first.

    Inputs: image - (224,224,3)
            filts -

    Outputs: angle (between 0 and 1) - remember to rescale later
    intensity = (between 0 and 1) - 0 represents no angle energy

    """

    perceptually_uniform = cspace_convert(image, "sRGB1", "JCh")
    lightness_image = perceptually_uniform[:, :, 0]

    magnitudes = []
    for filt in filts:
        sin_conv = convolve2d(lightness_image, filt[1], mode='same')
        cos_conv = convolve2d(lightness_image, filt[0], mode='same')

        magnitudes.append(np.sqrt(sin_conv ** 2 + cos_conv ** 2))

    orientation_vec = [magnitudes[0] - magnitudes[2],
                       magnitudes[1] - magnitudes[3]]
    angle = np.arctan2(magnitudes[0] - magnitudes[2],
                       magnitudes[1] - magnitudes[3])

    rescaled_angle = angle / (np.pi * 2) + 0.5

    intensity = np.sqrt(orientation_vec[0] ** 2 +
                        orientation_vec[1] ** 2) / \
                np.max(np.sqrt(orientation_vec[0] ** 2 +
                               orientation_vec[1] ** 2))

    return rescaled_angle, intensity


def get_all_orientations(val_dataset, kernel_size, output_path='./'):
    """Loops over all the images. For each image, saves a new image at `output_path`
    where the first channel is the edge intensity,
    the second channel is the degree of orientation
    and the third channel is all zeros."""
    ## run for all validation images and save to file

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1, shuffle=False,
                                             num_workers=1, pin_memory=True)

    output_folder = output_path + str(kernel_size)

    !mkdir - p $output_folder

    # define the gabor filters
    filts = get_quadratures(kernel_size)

    for i, (image, label) in tqdm(enumerate(val_loader)):
        # get first image in batch (of 1 anyways)
        image = image[0]
        image = np.squeeze(np.moveaxis(image.numpy(), 0, -1))

        angle, intensity = get_orientation_map(image, filts)

        image_to_save = np.stack((intensity, angle, np.zeros(angle.shape)), axis=2)

        name = "orientation_{:06d}.PNG".format(i)

        plt.imsave(output_folder + "/" + name, image_to_save)

        if i == 0:
            im = plt.imread(output_folder + "/" + name)[:, :, :2]
            print("angle rmse", np.mean(np.square(angle - im[:, :, 1])))
            print("intensity rmse", np.mean(np.square(intensity - im[:, :, 0])))


for kernel_size in (5,10,15,20,25,30):
    get_all_orientations(val_dataset_nonnormalized,
                         kernel_size,
                        '/home/abenjamin/DNN_illusions/data/features/orientations/PNGs/')

!ls / home / abenjamin / DNN_illusions / data / illusions








