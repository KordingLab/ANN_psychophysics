from data_generators import (single_black_line, get_quadratures,
                             get_orientation_map, add_random_line_color_n, add_random_line_aa_color_n,
                             add_random_polygon_perimeter_n, add_random_ellipse_perimeter, add_random_curved_lines,
                             add_random_line_aa_color_width_n, many_white_lines
                             )
import pandas as pd
import numpy as np

KERNEL_SIZE = 15
filts = get_quadratures(KERNEL_SIZE)



def gen_batch_thin_lines(red, green, blue, num_lines_per_img):
    all_inputs = list()
    all_targets = list()
    n_samples = 10000
    for n in range(n_samples):
        image_array = np.zeros((3, 224, 224))
        input = add_random_line_color_n(image_array, red, green, blue, num_lines_per_img)

        target = get_orientation_map(input[0], filts)

        all_inputs.append(input.reshape(-1))
        all_targets.append(target.reshape(-1))

    return all_inputs, all_targets


def gen_batch_thin_lines_aa(red, green, blue, num_lines_per_img):
    all_inputs = list()
    all_targets = list()
    n_samples = 10000
    for n in range(n_samples):
        image_array = np.zeros((3, 224, 224))
        input = add_random_line_aa_color_n(image_array, red, green, blue, num_lines_per_img)

        target = get_orientation_map(input[0], filts)

        all_inputs.append(input.reshape(-1))
        all_targets.append(target.reshape(-1))

    return all_inputs, all_targets


def gen_batch_lines_aa_width(red, green, blue, max_width, num_lines_per_img):
    all_inputs = list()
    all_targets = list()
    n_samples = 10000
    for n in range(n_samples):
        image_array = np.zeros((3, 224, 224))
        input = add_random_line_aa_color_width_n(image_array, red, green, blue, max_width, num_lines_per_img)

        target = get_orientation_map(input[0], filts)

        all_inputs.append(input.reshape(-1))
        all_targets.append(target.reshape(-1))

    return all_inputs, all_targets


def gen_batch_random_polygon_perimeter(red, green, blue, num_lines_per_img):
    all_inputs = list()
    all_targets = list()
    n_samples = 10000
    for n in range(n_samples):
        image_array = np.zeros((3, 224, 224))
        input = add_random_polygon_perimeter_n(image_array, red, green, blue, num_lines_per_img)

        target = get_orientation_map(input[0], filts)

        all_inputs.append(input.reshape(-1))
        all_targets.append(target.reshape(-1))

    return all_inputs, all_targets


def gen_batch_random_ellipse_perimeter(red, green, blue, max_width, num_lines_per_img):
    all_inputs = list()
    all_targets = list()
    n_samples = 10000
    for n in range(n_samples):
        image_array = np.zeros((3, 224, 224))
        input = add_random_ellipse_perimeter(image_array, red, green, blue, max_width, num_lines_per_img)

        target = get_orientation_map(input[0], filts)

        all_inputs.append(input.reshape(-1))
        all_targets.append(target.reshape(-1))

    return all_inputs, all_targets


def gen_batch_random_curved_lines(red, green, blue, num_lines_per_img):
    all_inputs = list()
    all_targets = list()
    n_samples = 10000
    for n in range(n_samples):
        image_array = np.zeros((3, 224, 224))
        input = add_random_curved_lines(image_array, red, green, blue, num_lines_per_img)

        target = get_orientation_map(input[0], filts)

        all_inputs.append(input.reshape(-1))
        all_targets.append(target.reshape(-1))

    return all_inputs, all_targets


def gen_many_lines_on_white():
    all_inputs = list()
    all_targets = list()
    n_samples = 20000
    for n in range(n_samples):
        image_array = np.zeros((3, 224, 224))
        input = many_white_lines(filts)

        all_inputs.append(input.reshape(-1))
        all_targets.append(target.reshape(-1))

    return all_inputs, all_targets

def main():


    all_inputs, all_targets = gen_many_lines_on_white()

    all_inputs = pd.DataFrame(np.stack(all_inputs))
    all_inputs.to_hdf('/home/abenjamin/DNN_illusions/fast_data/features/many_white_on_black/lines.h5', key="l", mode='w')


    all_targets = pd.DataFrame(np.stack(all_targets))
    all_targets.to_hdf('/home/abenjamin/DNN_illusions/fast_data/features/many_white_on_black/lines_targets.h5', key="l", mode='w')


main()





