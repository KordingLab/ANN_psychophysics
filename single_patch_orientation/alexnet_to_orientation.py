import numpy as np

import matplotlib.pyplot as plt
import argparse

import torch
import torch.optim as optim

import orientation_stim
import toy_model_train_orientation


"""
example of use:
    python alexnet_to_orientation.py --epochs 10 \
    --save-model --model-name 'alexNet_broadband_multiorinoise_naturaloriprior' \
    --if-more-noise-levels 
"""

def main():
    parser = argparse.ArgumentParser(description='Orientation with pre-trained alexnet')
    parser.add_argument('--input-img-size', type=int, default=224, metavar='N',
                        help='input image size (default: 224)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epoch-size', type=int, default=100, metavar='N',
                        help='epoch size (# of batches) for generating images online.')
    parser.add_argument('--test-epoch-size', type=int, default=20, metavar='N',
                        help='epoch size (# of batches) for generating images online.')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model-name', type=str, default='orient_cnn',
                        help='Name of the current Model for saving')
    parser.add_argument('--if-unif', action='store_true', default=False,  # use natural prior by default
                        help='if the training distribution uniform')
    parser.add_argument('--if-more-noise-levels', action='store_true', default=False,
                        help='if multiple orientation noise levels used in training')

    args = parser.parse_args()
    args.vis_name = args.model_name

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    model = toy_model_train_orientation.SlimAlexNet(max_pool_layer_index=1,
                                          last_layer_num_params=2).to(device)

    # the parameters would be finetuned - should only be conv_pool.5
    params_to_update = model.parameters()
    print("Params to learn:")
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

    optimizer_ft = optim.Adam(params_to_update, lr=1e-4)

    test_loss_history = []
    report_epoch = [1, 2, 4, 8, 16]
    for epoch in range(1, args.epochs + 1):
        toy_model_train_orientation.train(args, model, device,
                                          optimizer_ft, epoch, if_alexNet=True)
        test_loss = toy_model_train_orientation.test(args, model, device,
                                                     if_alexNet=True)
        test_loss_history.append(test_loss)
        # save intermediate result figures
        if epoch in report_epoch:
            if (args.save_model):
                torch.save(model.state_dict(),
                           args.model_name + '_epoch' + str(epoch) + '.pt')
                # save a figure with bias
                model_file = args.model_name + '_epoch' + str(epoch) + '.pt'
                ave_ori, all_ave_bias = toy_model_train_orientation.compare_bias(model_file,
                                                                                 img_size=224, if_alexNet=True)
                plt.figure(figsize=(5, 3))
                plt.plot(ave_ori, all_ave_bias[:, 1] - all_ave_bias[:, 0])
                plt.plot(ave_ori, all_ave_bias[:, 2] - all_ave_bias[:, 0])
                plt.plot([np.pi / 4, np.pi / 4], [-5 / 180 * np.pi, 5 / 180 * np.pi], 'k--')
                plt.plot([np.pi / 2, np.pi / 2], [-5 / 180 * np.pi, 5 / 180 * np.pi], 'k--')
                plt.plot([3 * np.pi / 4, 3 * np.pi / 4], [-5 / 180 * np.pi, 5 / 180 * np.pi], 'k--')
                plt.plot([0, np.pi], [0, 0], 'k--')
                plt.xlim([0, np.pi])
                plt.ylim(-5 / 180 * np.pi, 5 / 180 * np.pi)
                plt.savefig(args.vis_name + '_epoch' + str(epoch) + '.pdf')

    # save final report if not has been saved
    if args.epochs not in report_epoch:
        if (args.save_model):
            torch.save(model.state_dict(),
                       args.model_name + '_epoch' + str(epoch) + '.pt')

    # save a final figure with cosine similarity and bias
    feature_cosSim = toy_model_train_orientation.feature_similarity(args.input_img_size,
                                                   model, feature_layer_ind=3, if_alexNet=True)
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(179), feature_cosSim)
    plt.xlim([0, 180])
    # bias differences
    model_file = args.model_name + '_epoch' + str(epoch) + '.pt'
    ave_ori, all_ave_bias = toy_model_train_orientation.compare_bias(model_file,
                                                                     img_size=224, if_alexNet=True)
    plt.subplot(1, 2, 2)
    plt.plot(ave_ori, all_ave_bias[:, 1] - all_ave_bias[:, 0])
    plt.plot(ave_ori, all_ave_bias[:, 2] - all_ave_bias[:, 0])
    plt.plot([0, np.pi], [0, 0], 'k--')
    plt.plot([np.pi / 4, np.pi / 4], [-5 / 180 * np.pi, 5 / 180 * np.pi], 'k--')
    plt.plot([np.pi / 2, np.pi / 2], [-5 / 180 * np.pi, 5 / 180 * np.pi], 'k--')
    plt.plot([3 * np.pi / 4, 3 * np.pi / 4], [-5 / 180 * np.pi, 5 / 180 * np.pi], 'k--')
    plt.xlim([0, np.pi])
    plt.ylim(-5 / 180 * np.pi, 5 / 180 * np.pi)
    plt.savefig(args.vis_name + '_final_summary' + '.pdf')

    # figure showing the test loss progress
    plt.figure()
    plt.plot(np.arange(1, args.epochs + 1), test_loss_history)
    plt.savefig(args.vis_name + '_loss_history' + '.pdf')


if __name__ == '__main__':
    main()