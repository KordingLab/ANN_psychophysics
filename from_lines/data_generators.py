import torch
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import rotate
from skimage.draw import (line_aa, line, bezier_curve,polygon_perimeter, polygon,
                          ellipse, ellipse_perimeter)
import math


### define methods in torch to create the input data


def generate_shrunken_array_based_on_width(width):
    high_adjusted = 224 // width
    scaled_image = np.zeros((3, high_adjusted, high_adjusted))
    return scaled_image


def scale_array_by_width(img, width):
    scaled_image_red = np.repeat(np.repeat(img[0], width, axis=0), width, axis=1)
    scaled_image_green = np.repeat(np.repeat(img[1], width, axis=0), width, axis=1)
    scaled_image_blue = np.repeat(np.repeat(img[2], width, axis=0), width, axis=1)

    scaled_image = np.stack((scaled_image_red, scaled_image_green, scaled_image_blue))

    tuple_shape = scaled_image.shape

    print("Prior shape: ", tuple_shape)

    number_padding_row = 224 - tuple_shape[1]
    number_padding_column = 224 - tuple_shape[2]

    print(number_padding_row, number_padding_column)

    scaled_adjusted_image_red = np.pad(scaled_image[0], ((0, number_padding_row),
                                                         (0, number_padding_column)), mode='constant')
    scaled_adjusted_image_green = np.pad(scaled_image[1], ((0, number_padding_row),
                                                         (0, number_padding_column)), mode='constant')
    scaled_adjusted_image_blue = np.pad(scaled_image[2], ((0, number_padding_row),
                                                         (0, number_padding_column)), mode='constant')

    scaled_adjusted_image = np.stack((scaled_adjusted_image_red, scaled_adjusted_image_green,
                                                     scaled_adjusted_image_blue))

    print("Post shape: ", scaled_adjusted_image.shape)

    return scaled_adjusted_image



def draw_random_line_aa(im, red, green, blue):

    shape = im.shape
    row_length = shape[1]
    column_length = shape[2]

    r0 = np.random.randint(0, high=row_length)
    c0 = np.random.randint(0, high=column_length)

    r1 = np.random.randint(0, high=row_length)
    c1 = np.random.randint(0, high=column_length)

    rr, cc, val = line_aa(r0, c0, r1, c1)

    im[0, rr,cc] = val * red
    im[1, rr, cc] = val * green
    im[2, rr, cc] = val * blue

    return im

def draw_random_color_line(im, red, green, blue):

    shape = im.shape
    row_length = shape[1]
    column_length = shape[2]


    r0 = np.random.randint(0, high=row_length)
    c0 = np.random.randint(0, high=column_length)

    r1 = np.random.randint(0, high=row_length)
    c1 = np.random.randint(0, high=column_length)

    rr, cc = line(r0, c0, r1, c1)

    im[0, rr, cc] = red
    im[1, rr, cc] = green
    im[2, rr, cc] = blue

    return im


def check_outbounds(value):
    if (value < 0):
        return 0
    elif (value > 223):
        return 223

    return value


"""
This method draws a colored line perimeter give on a width. We compute two points randomly. Then, we take
the negative inverse slope and width distance to find two prime points relative to the first two points. Finally, we
use skimage draw line_aa four times with the four points to get a perimeter.

Important to note***
We return an array called "fake_img" that was previously empty so that we can easily fill in the perimeter if wanted 
(if there were multiple lines, the fill command becomes much more complicated). 
"""
def draw_colored_line_perimeter(im, red, green, blue, width):

    shape = im.shape
    row_length = shape[1]
    column_length = shape[2]

    r0 = np.random.randint(0, high=row_length)
    c0 = np.random.randint(0, high=column_length)

    r1 = np.random.randint(0, high=row_length)
    c1 = np.random.randint(0, high=column_length)

    c0_prime = None
    c1_prime = None

    r0_prime = None
    r1_prime = None

    if ((c0 - c1) == 0):
        #vertical line
        c0_prime = c0 + width
        c1_prime = c1+ width

        r0_prime = r0
        r1_prime = r1

    elif ((r0 - r1) == 0):
        #horizontal line

        c0_prime = c0
        c1_prime = c1

        r0_prime = r0 + width
        r1_prime = r1 + width

    else:

        m_slope = (r0 - r1) / (c0 - c1)

        c0_prime = math.floor((c0 * m_slope ** 2 + c0 - math.sqrt((m_slope ** 4 + m_slope ** 2) * width ** 2)) /
                              (m_slope ** 2 + 1))
        c1_prime = math.floor((c1 * m_slope ** 2 + c1 - math.sqrt((m_slope ** 4 + m_slope ** 2) * width ** 2)) /
                              (m_slope ** 2 + 1))

        r0_prime = math.floor(((c0_prime - c0) / m_slope) + r0)
        r1_prime = math.floor(((c1_prime - c1) / m_slope) + r1)

    c0_prime = int(check_outbounds(c0_prime))
    c1_prime = int(check_outbounds(c1_prime))

    r0_prime = int(check_outbounds(r0_prime))
    r1_prime = int(check_outbounds(r1_prime))


    rr11, cc11, val11 = line_aa(r0, c0, r1, c1)
    rr22, cc22, val22 = line_aa(r1, c1, r1_prime, c1_prime)
    rr33, cc33, val33 = line_aa(r1_prime, c1_prime, r0_prime, c0_prime)
    rr44, cc44, val44 = line_aa(r0_prime, c0_prime, r0, c0)

    #line 1
    im[0, rr11, cc11] = red * val11
    im[1, rr11, cc11] = green * val11
    im[2, rr11, cc11] = blue * val11
    #line 2
    im[0, rr22, cc22] = red * val22
    im[1, rr22, cc22] = green * val22
    im[2, rr22, cc22] = blue * val22
    #line3
    im[0, rr33, cc33] = red * val33
    im[1, rr33, cc33] = green * val33
    im[2, rr33, cc33] = blue * val33
    #line 4
    im[0, rr44, cc44] = red * val44
    im[1, rr44, cc44] = green * val44
    im[2, rr44, cc44] = blue * val44


    ###Fill Fake image up. Needed to compute perimeter fill
    fake_img = np.zeros((3, row_length, column_length))

    #line 1
    fake_img[0, rr11, cc11] = red * val11
    fake_img[1, rr11, cc11] = green * val11
    fake_img[2, rr11, cc11] = blue * val11
    #line 2
    fake_img[0, rr22, cc22] = red * val22
    fake_img[1, rr22, cc22] = green * val22
    fake_img[2, rr22, cc22] = blue * val22
    #line3
    fake_img[0, rr33, cc33] = red * val33
    fake_img[1, rr33, cc33] = green * val33
    fake_img[2, rr33, cc33] = blue * val33
    #line 4
    fake_img[0, rr44, cc44] = red * val44
    fake_img[1, rr44, cc44] = green * val44
    fake_img[2, rr44, cc44] = blue * val44

    return fake_img

"""
Fill perimeter takes in an array representing an image. Then, finds all the columns per a row that have nonzero values.
Thus, if we take a perimeter of a line, we know to fill in all the points in between with a color.
"""
def fill_perimeter(img):
    dimension_tuple = img.shape
    num_rows = dimension_tuple[1]
    num_columns = dimension_tuple[2]
    list_of_list= []
    for row in range(num_rows):
        count = 0
        bool_color = False
        list_x = []
        for column in range(1, num_columns):
            if (img[0][row][column] != 0):
                list_x.append(column)

        list_of_list.append(list_x)

    return list_of_list


def draw_color_line_width(im, red, green, blue, width):
    fake_img = draw_colored_line_perimeter(im, red, green, blue, width)
    list_of_list = fill_perimeter(fake_img)

    for index in range(len(list_of_list)):
        list_x = list_of_list[index]
        if (len(list_x) > 1):
            start = list_x[0]
            stop = list_x[len(list_x) - 1]
            im[0][index][start:stop] = red
            im[1][index][start:stop] = green
            im[2][index][start:stop] = blue


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

    shape = im.shape
    row_length = shape[1]
    column_length = shape[2]

    r0 = np.random.randint(0, high=row_length)
    c0 = np.random.randint(0, high=column_length)

    r1 = np.random.randint(0, high=row_length)
    c1 = np.random.randint(0, high=column_length)

    r2 = np.random.randint(0, high=row_length)
    c2 = np.random.randint(0, high=column_length)

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

    shape = im.shape
    row_length = shape[1]
    column_length = shape[2]

    for i in range(n_sides):
        x = np.random.randint(0, high=row_length)
        y = np.random.randint(0, high=column_length)

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


def draw_random_ellipse_perimeter_thick(im, red, green, blue, width):
    r = np.random.randint(0, high=224)
    c = np.random.randint(0, high=224)

    r_radius = np.random.uniform(0, 112)
    c_radius = np.random.uniform(0, 112)

    rotation = np.random.uniform(-np.pi, np.pi)

    rr, cc = ellipse(r, c, r_radius, c_radius, shape=im[0].shape, rotation=rotation)

    rr_small, cc_small = ellipse(r, c, r_radius - width, c_radius - width, shape = im[0].shape, rotation=rotation)
    im[0, rr, cc] = red
    im[1, rr, cc] = green
    im[2, rr, cc] = blue

    im[:, rr_small, cc_small] = 0


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


def add_random_line_aa_color_width_n(im, red, green, blue, max_width, n):
    for x in range(n):
        width = np.random.randint(0, high=max_width + 1)
        draw_color_line_width(im, red, green, blue, width)

    return im


def add_random_polygon_perimeter_n(im, red, green, blue, n):
    for x in range(n):
        draw_random_polygon_perimeter(im, red, green, blue)

    return im


def add_random_ellipse_perimeter(im, red, green, blue, max_width, n):
    for x in range(n):
        width= np.random.randint(0, high=max_width + 1)
        draw_random_ellipse_perimeter_thick(im, red, green, blue, width)

    return im


def add_random_curved_lines(im, red, green, blue, n):
    for x in range(n):
        draw_bezier_curve_line(im, red, green, blue)

    return im



#
# def main():
    # empty_img = np.zeros((3, 224, 224))
    # draw_color_line_width(empty_img, 100, 212, 10, 25)
    #
    # plt.imshow(empty_img[0])
    # plt.show()
    #
    # plt.imshow(empty_img[1])
    # plt.show()


# main()