### Import things
import torch
from torch.autograd import Variable
from data_loader_utils import data_iterator
from angle_decoder_linear import AngleDecoder

import pickle
import argparse
import os



def test(model, val_loader, val_target_loader, criterion, args, BATCH_SIZE):
    val_loss = 0
    model.eval()
    for batch_idx, (feats, orients) in enumerate(zip(val_loader, val_target_loader)):

        data, target = feats.cuda(), orients.cuda()
        data, target = Variable(data), Variable(target)
        data = (data - 0.456) / 0.224
        optimizer.zero_grad()
        output = model(data)
        if args.nonlinear:
            output = torch.mean(output,dim=0)
        loss = criterion(output, target)

        val_loss += loss.data

        if batch_idx * 36 > 360:
            break


    return val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--savename", help="prefix to save the model as (in /home/abenjamin/DNN_illusions/data/models/) ",
                        default='Angle_decoder',
                        type=str)
    parser.add_argument("layer", help="which layer of VGG the model was trained to decode from",
                        type=int)
    parser.add_argument("--epochs", help="how many epochs",
                        type=int, default=5)
    parser.add_argument("--n_filts", help="how filters around the center to decode from",
                        type=int, default=5)
    parser.add_argument("--noise-level", help="The variance of gaussian noise added to the vgg representation",
                        type=float, default=0)
    parser.add_argument("--stim-noise-level", help="The variance of gaussian noise added to the stimulus.",
                        type=float, default=0)
    parser.add_argument("--wd", help="Weight decay",
                        type=float, default=1e-5)
    parser.add_argument("--nonlinear", help="Use a 2-layer NN instead of a linear net on the decoder",
                        action='store_true')
    parser.add_argument("--init", help="Init with an old model",
                        action='store_true')
    parser.add_argument("--image_directory", type = str,
                        default='/home/abenjamin/DNN_illusions/fast_data/features/rel_angle/',
                        help="""Path to the folder in which we store the `lines.h5` and `lines_targets.h5` files.
                             If lines_targets.h5 does not exist, we just plot the input and model output.""")
    parser.add_argument("--card", help="which card to use",
                        type=int, default =0 )
    args = parser.parse_args()

    torch.manual_seed(0)

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.card)
    
    def load_model(model_path,layer,nonlinear,noise=0):
        """Loads the VGG+decoder network trained and saved at path."""
        model = AngleDecoder(layer, noise, nonlinear)

        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    ### define methods to load the data
    BATCH_SIZE = 50
    EPOCHS = args.epochs
    n_images = 2500

    samples = data_iterator(args.image_directory+'train/lines.h5', BATCH_SIZE, n_images)
    targets = data_iterator(args.image_directory+'train/lines_targets.h5', BATCH_SIZE, n_images)

    val_loader =  data_iterator(args.image_directory+'test/lines.h5', 36, 360)
    val_target_loader =  data_iterator(args.image_directory+'test/lines_targets.h5', 36, 360)


    ### define the network: pull from a conv2d layer of the pretrained vgg network and train on top of that
    vgg_and_decoder = AngleDecoder(args.layer, args.noise_level, args.nonlinear, args.n_filts)
    
    ###
    if args.init:
        # load a good old model
        model_path = "../../data/models/rel_angle_stimnoise_reprnoise-.01-layer_4.pt"
        old_model = load_model(model_path,4,False, noise=0)
        vgg_and_decoder.decoder[0].weight.data = old_model.decoder[0].weight.data * \
                                                torch.ones_like(vgg_and_decoder.decoder[0].weight)

    vgg_and_decoder = vgg_and_decoder.cuda()

    params_to_update = []
    for name,param in vgg_and_decoder.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    optimizer = torch.optim.Adam(params_to_update, 1e-4,
                                 weight_decay=args.wd)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=4, factor=.2, verbose=True)

    criterion = torch.nn.MSELoss()

    # Now loop through the data and train
    train_accuracy = []
    val_losses = []
    vgg_and_decoder.train()

    train_loss = 0
    i = 0
    # log_interval = 10
    old_train_loss=1e10
    done = False
    late = False
    for epoch in range(EPOCHS):
        if done:
            break
        for batch_idx, (feats, orients) in enumerate(zip(samples, targets)):
            vgg_and_decoder.train()

            data, target = feats.cuda(), orients.cuda()
            data, target = Variable(data), Variable(target)
            #approx normalize
            data = (data -  0.456)/ 0.224

            if args.stim_noise_level>0:
                data += torch.randn_like(data) * args.stim_noise_level

            optimizer.zero_grad()
            output = vgg_and_decoder(data)
            if args.nonlinear:
                # cast to a 100 dim target. so target.size()=[50,100]
                target = 50*torch.ones(BATCH_SIZE,500).cuda() * target
                output *= 50

            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.data
            i += data.size()[0]


            if batch_idx * BATCH_SIZE > n_images:

            # if (batch_idx > log_interval) and (batch_idx % log_interval == 0):
                train_loss /= i / BATCH_SIZE
                i = 0

                train_accuracy.append(train_loss.item())

                val_loss = test(vgg_and_decoder, val_loader, val_target_loader,
                                criterion, args, BATCH_SIZE)
                val_losses.append(val_loss)

                print('Epoch: {} Batch #: {} Train Loss: {:.6f}  Val loss {:.6f}'.format(
                    epoch + 1, batch_idx,
                    train_loss, val_loss))
                # early stopping
                if train_loss>old_train_loss:
                    if strikes >5:
                        done=True
                        break
                    else:
                        strikes +=1
                else:
                    strikes = 0
                    
                    
                old_train_loss = train_loss
                train_loss = 0
                break

        scheduler.step(loss)

    ### finally we save the network for later

    pickle.dump((train_accuracy,val_losses),
                open("/home/abenjamin/DNN_illusions/data/models/{}_{}.traintest".format(
                                args.savename, args.layer), "wb"))
    torch.save(vgg_and_decoder.state_dict(), "/home/abenjamin/DNN_illusions/data/models/{}_{}.pt".format(
                                    args.savename, args.layer))
    print("Saved")
