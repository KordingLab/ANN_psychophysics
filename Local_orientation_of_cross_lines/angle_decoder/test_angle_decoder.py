import torch
from torch.autograd import Variable

import argparse
import os
from data_loader_utils import data_iterator
from angle_decoder_linear import AngleDecoder

#visualize
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import matplotlib
from colorspacious import cspace_convert
import numpy as np
from matplotlib import pyplot as plt

def gen_test_lines(mask=None):
    """ Two intersecting lines of certain central angle and relative angle and position.

    """
    all_inputs = list()
    angles = list()
    n_samples = 180
    for n in range(n_samples):
        centerloc = (112,112)
        fixed_angle = 0
        relative_angle = n*np.pi/360 # up to 90 deg

        mask = np.zeros((224, 224)).astype(np.bool) if mask is None else mask

        numpy_im = generate_intersecting_rgb(centerloc, fixed_angle, relative_angle,
                                negative_mask=mask,
                                             linewidth=1)



        all_inputs.append(numpy_to_torch(numpy_im).view(-1))
        angles.append(relative_angle)

    return all_inputs, angles

def pass_test_images(model, samples, gpu = True, batch_size = 10):
    """This script loads a model, as specified by the path, tests some images, and displays the decoded images.

    Inputs:
        samples: a list of images

    Returns a list of tuples of (input_image (3x224x224), orientation_image (2x224x224),
                        target_orientation_image (2x224x224))
    """

    samples = torch.stack(samples)

    if gpu:
        feats = samples.cuda()
    data = Variable(samples)

    output = model(data).detach()
    if gpu:
        output = output.cpu()
    del data

    return output.numpy()

def load_model(args):
    """Loads the VGG+decoder network trained and saved at path."""
    model = AngleDecoder(args.layer, 0, args.nonlinear)

    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    return model


def save_and_visualize(outputs, targets):
    """Here we take decoded relative angles [range(180)] and plot them, along with what the angles actually were

    """



    #
    #
    # for i,(input,output,target) in enumerate(images):
    #
    #     check_sizes(input, output)
    #
    #     plt.figure(figsize=(15, 5))
    #     plt.subplot(131)
    #     plt.imshow(np.moveaxis(input,0,2))
    #     ax = plt.gca()
    #     ax.set_axis_off()
    #     ax.set_title("Input")
    #
    #     plt.subplot(132)
    #     ax2 = show_orientation_image(output)
    #     ax2.set_title("Decoded orientation")
    #
    #     try:#if target is not None:
    #         plt.subplot(133)
    #         ax3 = show_orientation_image(target)
    #         ax3.set_title("Target orientation")
    #     except:
    #         print("Recomputing orientation image with kernel size {}".format(kernel_size))
    #         filts = get_quadratures(kernel_size)
    #
    #         # note that we invert to get the map
    #         target = get_orientation_map(1-np.mean(check_on_float_scale(input),axis=0), filts)
    #         plt.subplot(133)
    #         ax3 = show_orientation_image(target)
    #         ax3.set_title("Target orientation")
    #
    #
    #     plt.tight_layout()
    #     if save:
    #         plt.savefig("Decoded_test_image_{}.png".format(i))
    #         #save just decoded png
    #         print(type(output), output.shape)
    #         output_img = convert_to_orientation_image(output)
    #         target_img = convert_to_orientation_image(target)
    #         # matplotlib.image.imsave('only_original_img_{}'.format(i), input)
    #         matplotlib.image.imsave('only_decoded_img_{}.png'.format(i), output_img)
    #         matplotlib.image.imsave('only_target_img_{}.png'.format(i), target_img)
    #     plt.show()
    #
    #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="relative path to the saved model",
                        type=str, default='/home/abenjamin/DNN_illusions/data/models/Angle_decoder_4.pt')
    parser.add_argument("layer", help="which layer of VGG the model was trained to decode from",
                        type=int)

    parser.add_argument('--no-cuda', action='store_true',
                    help='Disable CUDA')
    parser.add_argument("--card", help="which card to use",
                        type=int, default =0 )
    parser.add_argument('--nonlinear', action='store_true',
                        help='Use the decoder with nonlinear 2 layer network')
    args = parser.parse_args()
    args.gpu = not args.no_cuda

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.card)

    # note that right now the model is on the cpu
    model = load_model(args)
    if args.gpu:
        model = model.cuda()

    # get data
    unit_circle = np.zeros((224, 224)).astype(np.bool)
    for i in range(224):
        for j in range(224):
            if (i - 112) ** 2 + (j - 112) ** 2 >= 100 ** 2:
                unit_circle[i, j] = True
    samples, targets = gen_test_lines(unit_circle)

    outputs = pass_test_images(model, samples,args.gpu)

    save_and_visualize(outputs, targets)
