from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import argparse

from itertools import count
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import torchvision.models as models
from functools import reduce
from operator import mul

import orientation_stim


# import importlib
# importlib.reload(module_name)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # assume input 64*64
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20,
                      kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=20, out_channels=50,
                      kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.readout = nn.Sequential(
            nn.Dropout(),
            nn.Linear(13 * 13 * 50, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(500, 2)  # orthogonal components for orientations
        )

    def forward(self, x):
        x = self.features(x)
        # x = torch.flatten(x, 1)
        x = x.view(-1, 13 * 13 * 50)
        x = self.readout(x)

        return x


class Flatten(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x.view(-1, self.n)


class SlimAlexNet(nn.Module):

    """
    AlexNet with the classifier layers (fully-connected Linear layers) removed.
    """

    MAX_POOL_LAYERS = [3, 6, 13]
    # the shape of the MaxPool2D layer can be inspected using the following code
    # from torchsummary import summary
    # summary(torchvision.models.alexnet(pretrained=True), (3, 224, 224))
    MAX_POOL_OUT_SHAPE = [
        (64, 27, 27),
        (192, 13, 13),
        (256, 6, 6)
    ]

    def __init__(self, max_pool_layer_index=1, dropout_ratio=0.5, last_layer_num_params=2):
        """
        Construct a slim version of AlexNet by only reusing the first a few layers.
        The last layer maps the output of the MaxPool2D layer to 2/180 classes corresponding
        to 2-unit vector/180 degrees.
        alexnet: pretrained AlexNet model.
        max_pool_layer_index: Select which maximum pooling layer to use (1, 2 or 3).
        dropout_ratio: The dropout ratio/proportion for the Dropout layer.
        """

        assert 1 <= max_pool_layer_index <= 3, \
            f"Invalid max_pool_layer_index {max_pool_layer_index}"

        super().__init__()

        # extract the first a few layers, then combine it with two extra layers
        # (1) a flattening layer that returns an 1d array
        # (2) a fully connected linear layer
        layers_to_stack = self.MAX_POOL_LAYERS[max_pool_layer_index - 1]
        maxpool_out_shape = self.MAX_POOL_OUT_SHAPE[max_pool_layer_index - 1]

        num_params = reduce(mul, maxpool_out_shape)

        alexnet = models.alexnet(pretrained = True)
        # freeze the parameters of all layers
        for param in alexnet.parameters():
            param.requires_grad = False  # fix weights

        self.conv_pool = nn.Sequential(*(
                list(alexnet.features.children())[:layers_to_stack] +
                [
                    Flatten(num_params),
                    nn.Dropout(p=dropout_ratio),
                    nn.Linear(num_params, last_layer_num_params)
                ]
        ))

    def forward(self, x):
        return self.conv_pool(x)


def next_image_batch(img_size, batch_size, ori, ori_sigma=10/180*np.pi,
                     if_alexNet=False):
    """
    orientation images generated online,

    Args:
        ori: in radians
        ori_sigma: in radians

    """
    if np.max(np.array([ori_sigma]).shape) == 1:  # use default value
        ori_sigma = np.matlib.repmat(ori_sigma, batch_size, 1)
    # must be float32 array
    if if_alexNet:
        ori_imgs = np.empty([batch_size, 3, img_size, img_size], dtype=np.float32)
    else:
        ori_imgs = np.empty([batch_size, 1, img_size, img_size], dtype=np.float32)
    # spatial_freq = np.random.uniform(.2, 2, batch_size)  # total 5.3 degree patch, .2~6 cycle/degree
    # spatial_phase = np.random.uniform(0, 2*np.pi, batch_size)

    ori_code = np.float32(np.transpose(np.vstack((np.sin(ori * 2), np.cos(ori * 2)))))
    # *2 to utilize the full 2pi space

    for i_img in range(batch_size):
        # ---- broadband noise, same orientation noise level (fixed orient_sigma),
        #      sf is low pass (center_sf=0, sf_sigma=10)                            ----
        # ori_img = orientation_stim.broadband_noise(size=64, contrast=.5,
        #                           orientation=ori[i_img]) + .5  # range 0 to 1

        # ---- broadband noise, same orientation noise level,
        #      sf is band pass matched with the range sampled in gratings           ----
        ori_img = orientation_stim.broadband_noise(size=img_size, contrast=.5, orientation=ori[i_img],
                                                   if_band_pass=True, if_low_pass=False,
                                                   orient_sigma=ori_sigma[i_img]) + .5

        # ---- gratings, single freq per image, but randomly sampled for each image ----
        # ori_img = orientation_stim.grating(size=64, pixelsPerDegree=12, spatial_freq=spatial_freq[i_img],
        #                          spatial_phase=spatial_phase[i_img], orientation=ori[i_img],
        #                          contrast=.5) + .5

        if if_alexNet:
            preprocessFn = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]) #may not need to keep this though
            # transforms.Normalize:
            # input[channel] = (input[channel] - mean[channel]) / std[channel]
            ori_img3 = np.repeat(ori_img, 3).reshape((224, 224, 3)).transpose((2, 0, 1))
            ori_img3 = preprocessFn(torch.tensor(ori_img3))
            # plt.imshow(ori_img3[0, :, :]), plt.colorbar(), plt.show()

            ori_imgs[i_img, :, :, :] = ori_img3
        else:
            ori_imgs[i_img, 0, :, :] = ori_img  # combine the whole batch

    return ori_imgs, ori_code


def sample_natural_ori_prior(batch_size, if_unif):
    if if_unif:
        samples = np.random.uniform(0, np.pi, batch_size)
    else:
        samples = np.empty(batch_size)
        # p(theta) /propto 2-|sin(2theta)|, normalizing constant: 2pi-2
        ind = 0
        while ind < batch_size:
            theta = np.random.uniform(0, np.pi, 1)
            sample_p = np.random.uniform(0, 1, 1)
            if sample_p < 1-.5*np.abs(np.sin(2*theta)):
                samples[ind] = theta
                ind += 1

    return samples


def train(args, model, device, optimizer, epoch, if_alexNet=False):
    model.train()
    train_loss = 0
    loss_fn = nn.MSELoss()
    for batch_idx in range(args.epoch_size):
        # sample orientation
        ori = sample_natural_ori_prior(batch_size=args.batch_size,
                                       if_unif=args.if_unif)
        if args.if_more_noise_levels:
            ori_sigma = np.random.uniform(.5/180*np.pi, 39.5/180*np.pi, args.batch_size)
        else:
            ori_sigma = 10/180*np.pi
        data, target = next_image_batch(args.input_img_size, args.batch_size,
                                        ori=ori, ori_sigma=ori_sigma,
                                        if_alexNet=if_alexNet)
        # numpy -> tensor
        data, target = torch.from_numpy(data).to(device), \
                       torch.from_numpy(target).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), args.epoch_size * args.batch_size,
                       100. * batch_idx / args.epoch_size, loss.item()))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / args.epoch_size))


def test(args, model, device, if_alexNet=False):
    model.eval()
    loss_fn = nn.MSELoss()
    test_loss = 0
    with torch.no_grad():
        for i in range(args.test_epoch_size):
            # sample orientation
            ori = sample_natural_ori_prior(batch_size=args.batch_size, if_unif=True)
            ori_sigma = np.random.uniform(.5 / 180 * np.pi, 39.5 / 180 * np.pi, args.batch_size)
            data, target = next_image_batch(args.input_img_size, args.batch_size,
                                            ori=ori, ori_sigma=ori_sigma,
                                            if_alexNet=if_alexNet)
            # numpy -> tensor
            data, target = torch.from_numpy(data).to(device), \
                           torch.from_numpy(target).to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss

    test_loss /= args.test_epoch_size
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


def feature_similarity(img_size, model, feature_layer_ind, if_alexNet=False):
    model.eval()
    if if_alexNet:
        # for alexNet, early layers are not changed during training (directly from alexNet),
        # thus would be the same throughout epoch (not use "model")
        alexnet = models.alexnet(pretrained=True)
        feature_extractor = nn.Sequential(*list(alexnet.features.children())[:feature_layer_ind])
        # print(feature_extractor)
    else:
        feature_extractor = nn.Sequential(*list(model.features.children())[:feature_layer_ind])

    # pairwise compare
    pair_gen = zip(count(0), count(1))
    cosSimilarity_ave = np.empty((179, 1,))
    k_img_no = 10
    for i, j in list(islice(pair_gen, 0, 179)):
        #     print('%d'%i + ', ' + '%d'%j)
        ang1_ind = i  # in degree
        ang2_ind = j
        # orientation noise level in the images is as default (10/180*np.pi)
        imgs1, _ = next_image_batch(img_size, k_img_no,
                                    ori=np.matlib.repmat(ang1_ind/180*np.pi, k_img_no, 1),
                                    if_alexNet=if_alexNet)
        imgs2, _ = next_image_batch(img_size, k_img_no,
                                    ori=np.matlib.repmat(ang2_ind/180*np.pi, k_img_no, 1),
                                    if_alexNet=if_alexNet)
        features1 = feature_extractor(torch.tensor(imgs1))
        features2 = feature_extractor(torch.tensor(imgs2))
        cosSimilarity = np.empty((k_img_no, 1,))
        for img_ind in np.arange(k_img_no):
            # cosine similarity
            cosSimilarity[img_ind] = cosine_similarity(
                features1[img_ind, :, :, :].data.numpy().reshape(1, -1),
                features2[img_ind, :, :, :].data.numpy().reshape(1, -1)
            )
        cosSimilarity_ave[i] = np.mean(cosSimilarity)
    return cosSimilarity_ave


def angle_diff(ang1, ang2):  # ang1 - ang2, in radians
    return np.angle(np.exp(1j * ang1) / np.exp(1j * ang2))


def read_out_bias(img_size, model, device, ori, ori_sigma=10/180*np.pi, if_alexNet=False):
    # model.load_state_dict(torch.load('orient_cnn.pt', map_location='cpu'))
    model.eval()
    data, target = next_image_batch(img_size, len(ori), ori=ori, ori_sigma=ori_sigma,
                                    if_alexNet=if_alexNet)
    data = torch.from_numpy(data).to(device)
    output = model(data)
    output = output.detach().numpy()
    output_angle = np.arctan2(output[:, 0], output[:, 1])    # range [-pi, pi]
    # output_angle = (output_angle + 2 * np.pi) % (2 * np.pi)  # range [0 2pi]
    # output_angle /= 2
    bias = angle_diff(output_angle, ori*2) / 2 # output_angle - ori
    return bias  # in radians

    # ---- test how noise in sin/cos(theta) transfer into arctan2(sin,cos) ----
    # test_size = 10000
    # ori = np.random.uniform(0, np.pi, test_size)
    # ori_code = np.float32(np.transpose(np.vstack((np.sin(2*ori), np.cos(2*ori)))))
    # e = np.random.uniform(-.5, .5, (test_size, 2))
    # new_ori_code = ori_code + e
    # ori_hat = np.arctan2(new_ori_code[:, 0], new_ori_code[:, 1])
    # bias = angle_diff(ori_hat, ori * 2) / 2
    # plt.scatter(ori, bias, alpha=.2), plt.ylim(-.3, .3)
    # rolling_n = 500
    # sort_ind = np.argsort(ori)
    # ave_ori = pd.rolling_mean(np.sort(ori), rolling_n)
    # ave_bias = pd.rolling_mean(bias[sort_ind], rolling_n)
    # plt.plot(ave_ori, ave_bias, color='black')
    # plt.show()


def compare_bias(model_file, img_size=64, if_alexNet=False):
    # call "read_out_bias" with 3 levels of noise
    device = "cpu"
    if if_alexNet:
        model = SlimAlexNet(max_pool_layer_index=1,
                    last_layer_num_params=2).to(device)
    else:
        model = Net().to(device)

    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()
    rolling_n = 25
    # test images
    ori = np.random.uniform(0, np.pi, 1000)
    sort_ind = np.argsort(ori)
    sorted_ori = np.sort(ori)
    ave_ori = pd.Series(sorted_ori).rolling(rolling_n).mean()
    all_ori_sigma = np.array([3, 15, 27])/180*np.pi
    all_ave_bias = np.empty([ori.shape[0], all_ori_sigma.shape[0]])
    for ind, ori_sigma in enumerate(all_ori_sigma):
        bias = read_out_bias(img_size, model, device, ori, ori_sigma=ori_sigma,
                             if_alexNet=if_alexNet)
        all_ave_bias[:, ind] = pd.Series(bias[sort_ind]).rolling(rolling_n).mean()
    return ave_ori, all_ave_bias
    # plt.plot(ave_ori, all_ave_bias)
    # plt.plot([0, np.pi], [0, 0], 'k-')
    # plt.ylim(-5 / 180 * np.pi, 5 / 180 * np.pi)
    # plt.show()

    # ----           OR plot the difference              ----
    # plt.plot(ave_ori, all_ave_bias[:, 1] - all_ave_bias[:, 0])
    # plt.plot(ave_ori, all_ave_bias[:, 2] - all_ave_bias[:, 0], '--')
    # plt.plot([0, np.pi], [0, 0], 'k-.')
    # plt.ylim(-5 / 180 * np.pi, 5 / 180 * np.pi)
    # plt.show()


def result_figure_to_save(args, model, device, epoch):
    # save the model (.pt)
    if (args.save_model):
        torch.save(model.state_dict(),
                   args.model_name + '_epoch' + str(epoch) + '.pt')

    # feature representation similarity
    feature_cossim = feature_similarity(args.input_img_size, model,
                                        feature_layer_ind=3, if_alexNet=True)
    plt.figure(figsize=(10, 3))
    plt.subplot(1,3,1)
    plt.plot(np.arange(179), feature_cossim)
    plt.xlim([0, 180])
    # 2nd conv2d layer
    feature_cossim = feature_similarity(args.input_img_size, model,
                                        feature_layer_ind=6, if_alexNet=True)
    plt.subplot(1, 3, 2)
    plt.plot(np.arange(179), feature_cossim)
    plt.xlim([0, 180])

    # predict orientation bias
    ori = sample_natural_ori_prior(batch_size=args.batch_size * 10, if_unif=True)
    ori_sigma = np.random.uniform(.5/180*np.pi, 39.5/180*np.pi, args.batch_size * 10)
    bias = read_out_bias(args.input_img_size, model, device, ori=ori, ori_sigma=ori_sigma)  # in radians
    # rolling average
    rolling_n = 50
    sort_ind = np.argsort(ori)
    ave_ori = pd.Series(np.sort(ori)).rolling(rolling_n).mean()
    ave_bias = pd.Series(bias[sort_ind]).rolling(rolling_n).mean()
    plt.subplot(1,3,3)
    plt.scatter(ori/np.pi*180, bias/np.pi*180, alpha=.2)
    plt.xlim([0, 180]), plt.ylim([-20, 20])
    plt.plot(ave_ori/np.pi*180, ave_bias/np.pi*180, color='black')
    plt.plot([0, 180], [0, 0], '--', color='black')
    plt.plot([90, 90], [-20, 20], '--', color='black')
    plt.savefig(args.vis_name + '_epoch' + str(epoch) + '.pdf')
    # plt.show()

# class Args:
#     pass
# args = Args()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch orientation Example')
    parser.add_argument('--input-img-size', type=int, default=64, metavar='N',
                        help='input image size (default: 64)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epoch-size', type=int, default=100, metavar='N',
                        help='epoch size (# of batches) for generating images online.')
    parser.add_argument('--test-epoch-size', type=int, default=20, metavar='N',
                        help='epoch size (# of batches) for generating images online.')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model-name', type=str, default='orient_cnn',
                        help='Name of the current Model for saving')
    # parser.add_argument('--feature-layer-ind', type=int, default=3,
    #                     help='Select feature layer to visualize the representation distance')
    # parser.add_argument('--vis-name', type=str, default='orient_cnn.pdf',
    #                     help='image file name when save results visualization')
    parser.add_argument('--if-unif', action='store_true', default=False,  # use natural prior by default
                        help='if the training distribution uniform')
    parser.add_argument('--if-more-noise-levels', action='store_true', default=False,
                        help='if multiple orientation noise levels used in training')
    args = parser.parse_args()
    args.vis_name = args.model_name
    # example of use:
    # python toy_model_train_orientation.py --epochs 20 \
    # --save-model --model-name 'broadband_matchsf_multiorinoise_uniforiprior' \
    # --if-more-noise-levels --if-unif

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    test_loss_history = []
    report_epoch = [1, 2, 4, 8, 16]
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, optimizer, epoch)
        test_loss = test(args, model, device)
        test_loss_history.append(test_loss)
        # save intermediate result figures
        if epoch in report_epoch:
            result_figure_to_save(args, model, device, epoch)

    if args.epochs not in report_epoch:
        result_figure_to_save(args, model, device, args.epochs)  # final report

    plt.figure()
    plt.plot(np.arange(1, args.epochs + 1), test_loss_history)
    plt.savefig(args.vis_name + '_loss_history' + '.pdf')


if __name__ == '__main__':
    main()
