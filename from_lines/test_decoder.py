import torch
from torch.autograd import Variable

import argparse
import os
from decoder import OrientationDecoder
from data_loader_utils import data_iterator
from data_generators import get_quadratures, get_orientation_map
from decoder_nonlinear import OrientationDecoder as OrientationDecoderNonlinear
from decoder_upsample import OrientationDecoder as OrientationDecoderUpsample
from decoder_upsample_nonlinear import OrientationDecoder as OrientationDecoderUpsampleNonlinear

#visualize
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import matplotlib
from colorspacious import cspace_convert
import numpy as np
from matplotlib import pyplot as plt



def pass_test_images(model, image_path, gpu = True):
    """This script loads a model, as specified by the path, tests some images, and displays the decoded images.


    Returns a list of tuples of (input_image (3x224x224), orientation_image (2x224x224),
                        target_orientation_image (2x224x224))
    """

    BATCH_SIZE = 65

    samples = data_iterator(image_path+'lines.h5', BATCH_SIZE)
    try:
        targets = next(data_iterator(image_path+'lines_targets.h5', BATCH_SIZE))
    except:
        targets = [None] * BATCH_SIZE

    for batch_idx, feats in enumerate(samples):
        if gpu:
            feats = feats.cuda()
        data = Variable(feats)

        output = model(data).detach()
        if gpu:
            output = output.cpu()
            data = data.cpu()
        break

    # Right now we have three 4x_x224x224 tensors, and we want it in list of tuple form
    input_output_target = []
    for i in range(BATCH_SIZE):
        inp = data[i].numpy()
        out = output[i].numpy()
        tar = targets[i]
        input_output_target.append((inp,out,tar))

    return input_output_target



def save_and_visualize(images, kernel_size=15, save = True):
    """Iterates through images and saves and prints both the original images, output orientation images,
    and target orientation images, in that order.


    Images : a list of tuples of (input_image (3x224x224), orientation_image (2x224x224),
                        target_orientation_image (2x224x224))

    """
    for i,(input,output,target) in enumerate(images):

        check_sizes(input, output)

        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(np.moveaxis(input,0,2))
        ax = plt.gca()
        ax.set_axis_off()
        ax.set_title("Input")

        plt.subplot(132)
        ax2 = show_orientation_image(output)
        ax2.set_title("Decoded orientation")

        try:#if target is not None:
            plt.subplot(133)
            ax3 = show_orientation_image(target)
            ax3.set_title("Target orientation")
        except:
            print("Recomputing orientation image with kernel size {}".format(kernel_size))
            filts = get_quadratures(kernel_size)

            # note that we invert to get the map
            target = get_orientation_map(1-np.mean(check_on_float_scale(input),axis=0), filts)
            plt.subplot(133)
            ax3 = show_orientation_image(target)
            ax3.set_title("Target orientation")


        plt.tight_layout()
        if save:
            plt.savefig("Decoded_test_image_{}.png".format(i))
            #save just decoded png
            print(type(output), output.shape)
            output_img = convert_to_orientation_image(output)
            target_img = convert_to_orientation_image(target)
            # matplotlib.image.imsave('only_original_img_{}'.format(i), input)
            matplotlib.image.imsave('only_decoded_img_{}.png'.format(i), output_img)
            matplotlib.image.imsave('only_target_img_{}.png'.format(i), target_img)
        plt.show()

def check_on_float_scale(image):
    if np.any(image > 2.):
        image = image/255.
    return image

def check_sizes(input,output):
    assert input.shape == (3,224,224)
    assert output.shape == (2, 224, 224)

def get_uniform_colormap():
    color_circle = np.ones((256, 3)) * 66
    color_circle[:, 1] = np.ones((256)) * 44

    color_circle[:, 2] = np.arange(0, 360, 360 / 256)

    color_circle_rgb = cspace_convert(color_circle, "JCh", "sRGB1")
    cm = ListedColormap(color_circle_rgb)
    return cm

def convert_to_orientation_image(orientation_image, equiluminant = False):
    cmap = get_uniform_colormap()

    angle_image = np.arctan2(orientation_image[1], orientation_image[0], )/np.pi/2+0.5
    rgba_image = cmap(angle_image)
    #drop the alpha channel
    rgb_image = rgba_image[:,:,:3]


    #convert to hsl
    hsv_image = mpl.colors.rgb_to_hsv(rgb_image)

    # make v channel the magnitude
    magnitudes = np.maximum(np.minimum(np.nan_to_num(np.sqrt(np.square(orientation_image[0])
                                                                +np.square(orientation_image[1]) )),1),0)

    hsv_image[:,:,2] = magnitudes
    rgb_image = mpl.colors.hsv_to_rgb(hsv_image)

    return rgb_image

def show_orientation_image(orientation_image, equiluminant = False):
    """Takes a 2x224x224 image, with the two channels corresponding to x and y values,
    and returns a 224,224,3 image where we have converted to RGB, the hue of which
    represents phase and the brightness represents magnitude.

    This works by interpreting the x and y values at each pixel as giving the hue angle arctan2(y,x)
    in CIECAM02 color space.

    For dramatic effect, we also scale the brightness (L axis in CIELab) by the distance.
    This can be turned off by setting `equiluminant = True`"""
    cmap = get_uniform_colormap()

    angle_image = np.arctan2(orientation_image[1], orientation_image[0], )/np.pi/2+0.5
    rgba_image = cmap(angle_image)
    #drop the alpha channel
    rgb_image = rgba_image[:,:,:3]

    if not equiluminant:
        #convert to hsl
        hsv_image = mpl.colors.rgb_to_hsv(rgb_image)

        # make v channel the magnitude
        magnitudes = np.maximum(np.minimum(np.nan_to_num(np.sqrt(np.square(orientation_image[0])
                                                                +np.square(orientation_image[1]) )),1),0)

        hsv_image[:,:,2] = magnitudes
        rgb_image = mpl.colors.hsv_to_rgb(hsv_image)

    # just to get the colorbar
    a = plt.imshow((angle_image-0.5)*np.pi,
                   cmap=cmap,
                   vmin=-np.pi,
                   vmax=np.pi)

    plt.imshow(rgb_image)

    ax = plt.gca()
    ax.set_axis_off()
    cbar = plt.colorbar(a, aspect=10, fraction=.07)
    cbar.ax.set_ylabel('Phase [pi]')
    return ax



def load_model(args):
    """Loads the VGG+decoder network trained and saved at path."""
    if args.upsample:
        if args.nonlinear:
            model = OrientationDecoderUpsampleNonlinear(args.layer)
        else:
            model = OrientationDecoderUpsample(args.layer)
    else:
        if args.nonlinear:
            model = OrientationDecoderNonlinear(args.layer)
        else:
            model = OrientationDecoder(args.layer)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="relative path to the saved model",
                        type=str)
    parser.add_argument("layer", help="which layer of VGG the model was trained to decode from",
                        type=int)
    parser.add_argument("--image-directory", type = str,
                        default='/home/abenjamin/DNN_illusions/fast_data/features/straight_lines/',
                        help="""Path to the folder in which we store the `lines.h5` and `lines_targets.h5` files.
                             If lines_targets.h5 does not exist, we just plot the input and model output.""")
    parser.add_argument('--no-cuda', action='store_true',
                    help='Disable CUDA')
    parser.add_argument("--card", help="which card to use",
                        type=int, default =0 )
    parser.add_argument('--upsample', action='store_true',
                                help='Use the decoder with upsampling')
    parser.add_argument('--nonlinear', action='store_true',
                        help='Use the decoder with nonlinear 2 layer network')
    args = parser.parse_args()
    args.gpu = not args.no_cuda

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.card)

    # note that right now the model is on the cpu
    model = load_model(args)
    if args.gpu:
        model = model.cuda()
    images = pass_test_images(model, args.image_directory,args.gpu)

    save_and_visualize(images)
