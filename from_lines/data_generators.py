import torch
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import rotate
from skimage.draw import (line_aa, line, bezier_curve,polygon_perimeter, polygon,
                          ellipse, ellipse_perimeter)
import matplotlib.pyplot as plt



### define methods in torch to create the input data

def batch_inputs(filts, batchsize = 64):
    """Wraps the construction of the oriented lines, batches them, and returns pytorch tensors"""
    inputs,targets = list(zip(*[[torch.tensor(im, dtype = torch.float) for im in single_black_line(filts)]
                                       for _ in range(batchsize)]))

    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    assert inputs.size()[0] == batchsize

    return inputs,targets


def draw_random_line(im):
    """Takes a 3x224x224 image and adds to it a random oriented line"""

    r0 = np.random.randint(0, high=224)
    c0 = np.random.randint(0, high=224)

    r1 = np.random.randint(0, high=224)
    c1 = np.random.randint(0, high=224)

    rr, cc = line(r0, c0, r1, c1)

    im[:, rr,cc] = 255

    return im



def draw_random_line_aa(im, red, green, blue):

    r0 = np.random.randint(0, high=224)
    c0 = np.random.randint(0, high=224)

    r1 = np.random.randint(0, high=224)
    c1 = np.random.randint(0, high=224)

    rr, cc, val = line_aa(r0, c0, r1, c1)

    im[0, rr,cc] = val * red
    im[1, rr, cc] = val * green
    im[2, rr, cc] = val * blue

    return im

def draw_random_color_line(im, red, green, blue):

    r0 = np.random.randint(0, high=224)
    c0 = np.random.randint(0, high=224)

    r1 = np.random.randint(0, high=224)
    c1 = np.random.randint(0, high=224)

    rr, cc = line(r0, c0, r1, c1)

    im[0, rr, cc] = red
    im[1, rr, cc] = green
    im[2, rr, cc] = blue

    return im

def red_line_with_blue_distractors(filts, num_distractors = 10):
    """Routine for creating input and target images containing a straight red oriented line and with
    blue thin distracting lines the background. The `orientation image` target contains the orientation
    of just the red line, not the blue lines.

    Input: filts: an array of 4 gabor filters, precomputed for speed. Result of `get_quadratures`
    =====  halfwidth = width of the line in the returned array
    Returns a tuple:
    a) A numpy array (3, 224,224) containing a single straight red line with a random angle and position,
                                with many blue random lines in the background
    b) A numpy array (2,224,224) containing the orientation vector at every point in the above array
            ** for just the red line **.
            The first channel is "x" and the second channel is "y".
            Note that the angle at each point is np.arctan2(y,x)"""
    raise NotImplementedError("")


def draw_bezier_curve_line(im, red, green, blue):

    r0 = np.random.randint(0, high=224)
    c0 = np.random.randint(0, high=224)

    r1 = np.random.randint(0, high=224)
    c1 = np.random.randint(0, high=224)

    r2 = np.random.randint(0, high=224)
    c2 = np.random.randint(0, high=224)

    weight = np.random.randint(0, 11)

    rr, cc = bezier_curve(r0, c0, r1, c1, r2, c2, weight, (224, 224))

    im[0, rr, cc] = red # im[0][rr,cc]?
    im[1, rr, cc] = green
    im[2, rr, cc] = blue

    return im

def draw_random_polygon_solid(im, red, green, blue, n_sides):

    x_array = []
    y_array = []

    for i in range(n_sides):
        x = np.random.randint(0, high=224)
        y = np.random.randint(0, high=224)

        x_array.append(x)
        y_array.append(y)



    rr, cc = polygon(x_array, y_array)
    im[0, rr, cc] = red
    im[1, rr, cc] = green
    im[2, rr, cc] = blue

    return im


def draw_random_polygon_perimeter(im, red, green, blue, n_sides):
    x_array = []
    y_array = []

    for i in range(n_sides):
        x = np.random.randint(0, high=224)
        y = np.random.randint(0, high=224)

        x_array.append(x)
        y_array.append(y)


    rr, cc = polygon_perimeter(x_array, y_array, shape=im[0].shape, clip=True)
    im[0, rr, cc] = red
    im[1, rr, cc] = green
    im[2, rr, cc] = blue

    return im


def draw_random_ellipse_solid(im, red, green, blue):
        r = np.random.randint(0, high=224)
        c = np.random.randint(0, high=224)

        r_radius = np.random.uniform(0, 112)
        c_radius = np.random.uniform(0, 112)

        rotation = np.random.uniform(-np.pi, np.pi)

        rr, cc = ellipse(r,c,r_radius, c_radius, shape=im[0].shape, rotation=rotation)
        im[0, rr, cc] = red
        im[1, rr, cc] = green
        im[2, rr, cc] = blue


def draw_random_ellipse_perimeter(im, red, green, blue):
    r = np.random.randint(0, high=224)
    c = np.random.randint(0, high=224)

    r_radius = np.random.randint(0, high=112)
    c_radius = np.random.randint(0, high=112)

    orientation = np.random.uniform(-np.pi, np.pi)

    rr, cc = ellipse_perimeter(r, c, r_radius, c_radius, orientation = orientation, shape = im[0].shape)
    im[0, rr, cc] = red
    im[1, rr, cc] = green
    im[2, rr, cc] = blue


def curved_black_line(filts):
    """Routine for creating input and target images containing a curved black line.

    Input: filts: an array of 4 gabor filters, precomputed for speed. Result of `get_quadratures`
    =====  halfwidth = width of the line in the returned array

    Returns a tuple:
    =====  a) A numpy array (3, 224,224) containing a single curved line with a random curvature and position
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
    a) A numpy array (3, 224,224) containing a single straight line with a random angle and position
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


def add_random_line_color_n(im, red, green, blue, n):
    for x in range(n):
        draw_random_color_line(im,red, green, blue)

    return im

def add_random_line_aa_color_n(im, red, green, blue, n):
    for x in range(n):
        draw_random_line_aa(im, red, green, blue)

    return im

def add_random_polygon_perimeter_n(im, red, green, blue, n):
    for x in range(n):
        draw_random_polygon_perimeter(im, red, green, blue)

    return im

def add_random_ellipse_perimeter(im, red, green, blue, n):
    for x in range(n):
        draw_random_ellipse_perimeter(im, red, green, blue)

def main():
    empty_img = np.zeros((3, 224, 224))

    # print(empty_img)

    # add_random_line(empty_img)

    # bezier_curve_line(empty_img, 244, 244, 244)
    # add_random_line_color_n(empty_img, 255, 255, 1, 100)

    # add_test_line(empty_img)

    print(empty_img.shape)
    #
    # add_random_line_aa(empty_img)
    # random_polygon(empty_img, 200, 100, 55, 4)

    # for row in range(244):
    #     for col in range(244):
    #         value = empty_img[0, row, col]
    #         if (value != 0):
    #             print (row, col)

    draw_random_ellipse_perimeter(empty_img, 200, 100, 55)

    KERNEL_SIZE = 15
    filts = get_quadratures(KERNEL_SIZE)

    empty_img = get_orientation_map(empty_img[0], filts)

    plt.imshow(empty_img[0])
    plt.show()

    plt.imshow(empty_img[1])
    plt.show()

    plt.imshow(empty_img[2])
    plt.show()

main()