
import pandas as pd
import numpy as np
from scipy.ndimage import filters

from calculate_fisher_improved import generate_intersecting_rgb, numpy_to_torch


import argparse
import os
from tqdm import tqdm as tqdm

def gen_many_lines_on_white(noise=True, mask=None,N=1000, test = False):
    """ Two intersecting lines of random central angle and relative angle and position.

    if test = True, then we sample from angles with final digits in the range 0-1
    else the rest.


    """




    all_inputs = list()
    angles = list()
    n_samples = N


    if test:
        def gen_angle():
            tens = np.random.randint(0,18)*10
            rest = np.random.uniform(0,1)
            return (tens+rest)/180*np.pi-np.pi
    else:
        def gen_angle():
            tens = np.random.randint(0,18)*10
            rest = np.random.uniform(1,10)
            return (tens+rest)/180*np.pi-np.pi

    for n in tqdm(range(n_samples)):
        #at center
        centerloc = (112,112)#np.random.randint(80, 224 - 80, 2)
        fixed_angle = np.pi/2
        relative_angle = gen_angle()

        mask = np.zeros((224, 224)).astype(np.bool) if mask is None else mask

        numpy_im = generate_intersecting_rgb(centerloc, fixed_angle, relative_angle,
                                negative_mask=mask,
                                             linewidth=1)

        if noise:
            # blur by a random amount
            sigma = np.random.uniform(.1, 5, 3)
            im = filters.gaussian_filter(numpy_im, sigma=sigma)

            # add gaussian noise
            n = np.random.randn() * .1
            im = im + np.random.randn(*im.shape) * n

        all_inputs.append(numpy_to_torch(numpy_im).view(-1))
        angles.append(relative_angle)

    return all_inputs, angles


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--noise", help="Whether to add gaussian noise added to the raw images",
                        action = 'store_true')
    parser.add_argument("--n_images", help="How many to build",
                        type=int, default = 10000)
    parser.add_argument("--test", help="Test set or train set?",
                        action = 'store_true')
    parser.add_argument("--image_directory", type = str,
                        default='/home/abenjamin/DNN_illusions/fast_data/features/rel_angle/',
                        help="""Path to the folder in which we store the `lines.h5` and `lines_targets.h5` files.
                             If lines_targets.h5 does not exist, we just plot the input and model output.""")

    args = parser.parse_args()

    unit_circle = np.zeros((224, 224)).astype(np.bool)
    for i in range(224):
        for j in range(224):
            if (i - 112) ** 2 + (j - 112) ** 2 >= 100 ** 2:
                unit_circle[i, j] = True

    all_inputs, all_targets = gen_many_lines_on_white(args.noise, unit_circle, N=args.n_images, test = args.test)

    all_inputs = pd.DataFrame(np.stack(all_inputs))
    all_inputs.to_hdf(args.image_directory+'lines.h5', key="l", mode='w')


    all_targets = pd.DataFrame(np.stack(all_targets))
    all_targets.to_hdf(args.image_directory+'lines_targets.h5', key="l", mode='w')




