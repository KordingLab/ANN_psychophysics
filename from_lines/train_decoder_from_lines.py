### Import things
import torch
from torch.autograd import Variable
from data_loader_utils import data_iterator
from decoder import OrientationDecoder

#
savename = "Linear_decoder_plain_lines"

### define methods to load the data
BATCH_SIZE = 64
EPOCHS = 5
samples = data_iterator('/home/abenjamin/DNN_illusions/fast_data/features/straight_lines/lines.h5', BATCH_SIZE)
targets = data_iterator('/home/abenjamin/DNN_illusions/fast_data/features/straight_lines/lines_targets.h5', BATCH_SIZE)

### set which layer we'll decode from
LAYER = 4


### define the network: pull from a conv2d layer of the pretrained vgg network and train on top of that

vgg_and_decoder = OrientationDecoder(LAYER).cuda()

params_to_update = []
for name,param in vgg_and_decoder.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

optimizer = torch.optim.Adam(params_to_update, 1e-4,
                             weight_decay=1e-4)

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

        loss = criterion(output, target)
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

import pickle
pickle.dump(train_accuracy,
            open("/home/abenjamin/DNN_illusions/data/models/{}.traintest".format(savename), "wb"))
torch.save(vgg_and_decoder.state_dict(), "/home/abenjamin/DNN_illusions/data/models/{}.pt".format(savename))
print("Saved")
