import torch
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import rotate



### define methods in torch to create the input data

def batch_inputs(filts, batchsize = 64):
    """Wraps the construction of the oriented lines, batches them, and returns pytorch tensors"""
    inputs,targets = list(zip(*[[torch.tensor(im, dtype = torch.float) for im in single_black_line(filts)]
                                       for _ in range(batchsize)]))

    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    assert inputs.size()[0] == batchsize

    return inputs,targets


def _add_random_line(im):
    """Takes a 3x224x224 image and adds to it a random blue oriented line"""
    raise NotImplementedError("This will be a helper method for the `with_distractor` methods.")


def red_line_with_blue_distractors(filts, num_distractors = 10):
    """Routine for creating input and target images containing a straight red oriented line and with
    blue thin distracting lines the background. The `orientation image` target contains the orientation
    of just the red line, not the blue lines.

    Input: filts: an array of 4 gabor filters, precomputed for speed. Result of `get_quadratures`
    =====  halfwidth = width of the line in the returned array
    Returns a tuple:
    a) A numpy array (224,224) containing a single straight red line with a random angle and position,
                                with many blue random lines in the background
    b) A numpy array (2,224,224) containing the orientation vector at every point in the above array
            ** for just the red line **.
            The first channel is "x" and the second channel is "y".
            Note that the angle at each point is np.arctan2(y,x)"""
    raise NotImplementedError("")



def curved_black_line(filts):
    """Routine for creating input and target images containing a curved black line.

    Input: filts: an array of 4 gabor filters, precomputed for speed. Result of `get_quadratures`
    =====  halfwidth = width of the line in the returned array

    Returns a tuple:
    =====  a) A numpy array (224,224) containing a single curved line with a random curvature and position
           b) A numpy array (2,224,224) containing the orientation vector at every point in the above array.
            The first channel is "x" and the second channel is "y".
            Note that the angle at each point is np.arctan2(y,x)"""
    raise NotImplementedError("")



def single_black_line(filts, halfwidth = 1):
    """
    Routine for creating input and target images containing a straight oriented line

    Input: filts: an array of 4 gabor filters, precomputed for speed. Result of `get_quadratures`
    =====  halfwidth = width of the line in the returned array
    Returns a tuple:
    a) A numpy array (224,224) containing a single straight line with a random angle and position
    b) A numpy array (2,224,224) containing the orientation vector at every point in the above array.
            The first channel is "x" and the second channel is "y".
            Note that the angle at each point is np.arctan2(y,x)


    """
    angle = np.random.uniform(-np.pi, np.pi)
    centerloc = np.random.randint(0, 224, 2)

    starter = np.zeros((3,224 * 4, 224 * 4))
    x = np.cos(angle)
    y = np.sin(angle)

    # start with horizontal bar at center
    starter[:, (112 * 4 - halfwidth):(112 * 4 + halfwidth)] = 1
    # Rotate it
    im = rotate(starter, -angle * 180 / np.pi, axes=(1,2), reshape=False)
    im = im[:,224 + centerloc[0]:224 * 2 + centerloc[0], 224 + centerloc[1]:224 * 2 + centerloc[1]]

    orientation_image  = get_orientation_map(im[0],filts)

    return im,orientation_image




def get_orientation_map(image, filts):
    """Convolves 4 quadrature filters over an image and returns the image orientation.
    Converts to greyscale first.

    Inputs: image - (224,224) numpy array
            kernel size - size of the kernel of the gabors we'll convolve

    Outputs: (2,224,224) array, with first channel the x of the orientation vectors at each
                                pixel and the second channel the y's

    """



    # convolve Gabors and get energy of each
    magnitudes = []
    for filt in filts:
        sin_conv = convolve2d(image, filt[1], mode='same')
        cos_conv = convolve2d(image, filt[0], mode='same')

        magnitudes.append(np.sqrt(sin_conv ** 2 + cos_conv ** 2))

    orientation_vec = np.array([magnitudes[0] - magnitudes[2],
                                magnitudes[1] - magnitudes[3]])

    return orientation_vec


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
