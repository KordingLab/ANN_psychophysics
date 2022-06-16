import numpy as np
import scipy.ndimage
import matplotlib
from numpy import matlib

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# torch package
import torch
from torch.distributions import normal


def grating(size=500, pixelsPerDegree=200, spatial_freq=3, spatial_phase=0,
            orientation=np.pi/4, contrast=1):
    '''
    The output range is -1 to 1 if contrast is 1

    size: number of pixel of the image patch, assuming square
    spatial_freq: cycle per visual angle
    spatial_phase: in radians
    pixelsPerDegree: number of patch pixels in one degree visual angle
    orientation: in radians
    '''
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    x = (x - size / 2.0) / pixelsPerDegree
    y = (y - size / 2.0) / pixelsPerDegree

    return contrast * np.cos(spatial_phase +
                      2 * np.pi * spatial_freq *
                      (x * np.sin(orientation) + y * np.cos(orientation)))


def gabor(size=500, pixelsPerDegree=100, spatial_freq=3, spatial_phase=0,
          orientation=np.pi/4, contrast=1, sigma=.5, spatial_aspect_ratio=1):
    '''
    adds an exponential modulation (Gaussian envelope) on top of "grating"

    spatial_freq: cycle/visual angle
    spatial_phase: in radians
    orientation: in radians
    contrast: if 1, values range from -1 to 1
    sigma: standard deviation of the Gaussian envelope (in visual angle)
    spatial_aspect_ratio (gamma): if not 1, distorted along orientation.
           specifies the ellipticity of the support of the Gabor function.

    '''

    x, y = np.meshgrid(np.arange(size), np.arange(size))
    x = (x - size / 2.0) / pixelsPerDegree #in visual angle
    y = (y - size / 2.0) / pixelsPerDegree

    # rotation
    x_theta = x * np.cos(orientation) + y * np.sin(orientation)
    y_theta = -x * np.sin(orientation) + y * np.cos(orientation)

    # Gaussian envelope
    sigma_x = sigma
    sigma_y = float(sigma) / spatial_aspect_ratio
    gaussian_envelope = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 +
                                      y_theta ** 2 / sigma_y ** 2))

    # sinusoidal grating
    grating = np.cos(2 * np.pi * spatial_freq * x_theta + spatial_phase)
    gabor = gaussian_envelope * grating

    # normalize for contrast
    gabor_min = np.min(gabor[:])
    gabor_max = np.max(gabor[:])
    gabor = (gabor - gabor_min) * contrast * 2 / (gabor_max - gabor_min) - 1

    return gabor


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma))
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def circular_mask(size=500, pixelsPerDegree=200, radius=1, polarity_in=1, polarity_out=0,
                  if_filtered=False, filter_size = (15, 15), filter_width = 2):
    '''
    :param size: size of the image patch
    :param radius: in visual angle
    :param polarity_in: 1 or 0 inside the circle
    :param filter_size and filter_width are in pixel units
    :return: the mask
    '''
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    x = (x - size / 2.0) / pixelsPerDegree
    y = (y - size / 2.0) / pixelsPerDegree

    mask = np.ones([size, size]) * polarity_out
    mask[np.sqrt(np.power(x, 2) + np.power(y, 2)) < radius] = polarity_in

    # Gaussian filtering the mask
    if if_filtered:
        H = matlab_style_gauss2D(filter_size, filter_width)  # lowpass filter
        mask = scipy.ndimage.convolve(mask, H, mode='nearest')  # filter

    return mask

# --- example grating ---
# img = grating(500, 200, 3, np.pi/4, 1, np.pi/5)
# mask = circular_mask(500, 200, 1,
#                      if_filtered=True, filter_size=(50, 50), filter_width=10)
# img = np.multiply(img, mask)
# plt.imshow(img, cmap=plt.gray()), plt.show()


def broadband_noise(size=64, contrast=1, if_low_pass=True, center_sf=0, sf_sigma=10,
                    if_band_pass=False, low_sf=1.67, high_sf=10.67,
                    orientation=20/180*np.pi, orient_sigma=10/180*np.pi):
    """
    broadband noise with either low pass [--low-pass] or
                                band pass [--band-pass] spatial frequency;
                                if set one as true, need to make sure the other is false

    Args:
        size: if size is not power of 2, crop to be the intended size
        contrast:
        center_sf: if 0, low pass
        sf_sigma: in pixel; if Inf, all sf included (but exclude the corner)
        low_sf: cycle/image
        high_sf: cycle/image
        orientation: in radians, note that the available range of orientation is 0 - pi
        orient_sigma: in radians; if Inf, all orientations are included

    Return:
    """

    size_tmp = np.power(2, np.ceil(np.log2(size))).astype(int)  # then crop to intended size
    input_img = np.random.uniform(0, 1, [size_tmp, size_tmp])
    max_sf = size_tmp / 2
    img_center = matlib.repmat(np.floor(size_tmp / 2), 1, 2)

    # Fourier transform and separate magnitude and phase
    f = np.fft.fftshift(np.fft.fft2(input_img))
    # mag_f = np.abs(f)
    # phase_f = np.angle(f)

    # make generic matrices, where r represents frequency,
    # theta for orientation
    x, y = np.meshgrid(np.arange(-size_tmp/2, size_tmp/2),
                       np.arange(-size_tmp/2, size_tmp/2))
    r = np.sqrt(np.power(x, 2) + np.power(y, 2))
    y[y == 0] = .01
    theta = np.arctan(x / y)
    theta[f.shape[1] // 2:, :] = theta[f.shape[1] // 2 :, :] - np.pi
    theta += 3 * np.pi / 2
    # theta = np.arctan2(y, x) + np.pi  #in radians

    # build the filter, spatial freq filters
    if if_low_pass:
        if np.isinf(sf_sigma):
            sf_band = np.zeros(r.shape)
            sf_band[r <= max_sf] = 1
        else:
            sf_band = np.exp(-((r - center_sf) ** 2 / 2 / sf_sigma ** 2))
    elif if_band_pass:
        sf_band = np.zeros(r.shape)
        sf_band[(r >= low_sf) & (r<= high_sf)] = 1


    # orientation filters
    if np.isinf(orient_sigma):
        pass_band = sf_band
    else:  # need to make sure it works properly for 0-180 deg
        orient_band = np.exp(-((theta - (orientation + np.pi / 2)) ** 2 / 2 / orient_sigma ** 2)) + \
                      np.exp(-((np.angle(np.exp(1j * (theta))) + np.pi - (orientation + np.pi / 2)) ** 2
                               / 2 / orient_sigma ** 2))
        pass_band = np.multiply(orient_band, sf_band)
    # only pass the needed components, and reconstruct the image back
    band = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.multiply(f, pass_band))))
    band = np.real(band)

    # normalize for contrast
    band_min = np.min(band[:])
    band_max = np.max(band[:])
    band = (band - band_min) * contrast * 2 / (band_max - band_min) - 1

    # crop if needed
    if size_tmp > size:
        band = band[0:size, 0:size]

    return band

