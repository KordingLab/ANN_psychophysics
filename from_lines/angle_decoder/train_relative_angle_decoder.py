### Import things
import torch
from torch.autograd import Variable
from data_loader_utils import data_iterator
from angle_decoder_linear import AngleDecoder

import pickle
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--savename", help="prefix to save the model as (in /home/abenjamin/DNN_illusions/data/models/) ",
                        default='Angle_decoder',
                        type=str)
    parser.add_argument("layer", help="which layer of VGG the model was trained to decode from",
                        type=int)
    parser.add_argument("--epochs", help="how many epochs",
                        type=int, default=5)
    parser.add_argument("--noise-level", help="The variance of gaussian noise added to the vgg representation",
                        type=float, default=0)
    parser.add_argument("--nonlinear", help="Use a 2-layer NN instead of a linear net on the decoder",
                        action='store_true')
    parser.add_argument("--image_directory", type = str,
                        default='/home/abenjamin/DNN_illusions/fast_data/features/straight_lines/',
                        help="""Path to the folder in which we store the `lines.h5` and `lines_targets.h5` files.
                             If lines_targets.h5 does not exist, we just plot the input and model output.""")
    parser.add_argument("--card", help="which card to use",
                        type=int, default =0 )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.card)

    ### define methods to load the data
    BATCH_SIZE = 64
    EPOCHS = args.epochs
    samples = data_iterator(args.image_directory+'lines.h5', BATCH_SIZE)
    targets = data_iterator(args.image_directory+'lines_targets.h5', BATCH_SIZE)


    ### define the network: pull from a conv2d layer of the pretrained vgg network and train on top of that
    vgg_and_decoder = AngleDecoder(args.layer, args.noise_level, args.nonlinear).cuda()


    params_to_update = []
    for name,param in vgg_and_decoder.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    optimizer = torch.optim.Adam(params_to_update, 1e-4,
                                 weight_decay=1e-5)

    criterion = torch.nn.MSELoss()

    # Now loop through the data and train
    train_accuracy = []
    vgg_and_decoder.train()

    train_loss = 0
    i = 0
    log_interval = 10
    for epoch in range(EPOCHS):
        for batch_idx, (feats, orients) in enumerate(zip(samples, targets)):

            data, target = feats.cuda(), orients.cuda()
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output = vgg_and_decoder(data)

            #turn to vectors so we can do MSE
            loss = torch.mean((torch.sin(target) - torch.sin(output))**2 + \
                              (torch.cos(target) - torch.cos(output))**2)

            loss.backward()

            train_loss += loss.data
            i += data.size()[0]

            optimizer.step()

            if batch_idx * BATCH_SIZE > 10000:
                break

            if (batch_idx > log_interval) and (batch_idx % log_interval == 0):
                train_loss /= i / BATCH_SIZE
                i = 0

                train_accuracy.append(train_loss)

                print('Epoch: {} Batch #: {} Train Loss: {:.6f} '.format(
                    epoch + 1, batch_idx,
                    train_loss))
                train_loss = 0




    ### finally we save the network for later

    pickle.dump(train_accuracy,
                open("/home/abenjamin/DNN_illusions/data/models/{}_{}.traintest".format(
                                args.savename, args.layer), "wb"))
    torch.save(vgg_and_decoder.state_dict(), "/home/abenjamin/DNN_illusions/data/models/{}_{}.pt".format(
                                    args.savename, args.layer))
    print("Saved")
