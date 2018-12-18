
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import pickle

from model import Decoder
from data_utils import features_dataset, orientations_iterator
from train_utils import train


# orientation kernel size
KERNEL_SIZE = 30
assert KERNEL_SIZE in (5,10,15,20,25,30)
# which features  to take
LAYER_NUMBER = 24
assert LAYER_NUMBER in (24,31)

TEST_BATCH_SIZE = 64
BATCH_SIZE = 128
LOG_INTERVAL = 5
EPOCHS = 50

savename = "trainedmodel_kernel_{}_layer_{}".format(KERNEL_SIZE,LAYER_NUMBER)

feature_num_dict = {24: 100352, 31 :512*7*7}
num_features = feature_num_dict[LAYER_NUMBER]


# load the data
features_data_train = features_dataset('/home/abenjamin/DNN_illusions/fast_data/features/vgg_maxpool{}.h5'.format(LAYER_NUMBER), train = True)
features_data_valid = features_dataset('/home/abenjamin/DNN_illusions/fast_data/features/vgg_maxpool{}.h5'.format(LAYER_NUMBER), train = False)

model = Decoder(num_features).cuda()

TEST_BATCH_SIZE = 64
BATCH_SIZE = 128
LOG_INTERVAL = 5
EPOCHS = 10

features_loader = torch.utils.data.DataLoader(features_data_train,
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True, drop_last = True)

orientation_loader = orientations_iterator('/home/abenjamin/DNN_illusions/data/features/orientations/{}/all_images.h5'.format(KERNEL_SIZE),
                                                      BATCH_SIZE, train = True)


valid_features_loader = torch.utils.data.DataLoader(features_data_valid,
    batch_size=TEST_BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True, drop_last = True)

valid_orientation_loader = orientations_iterator('/home/abenjamin/DNN_illusions/data/features/orientations/{}/all_images.h5'.format(KERNEL_SIZE),
                                                      TEST_BATCH_SIZE, train = False)

optimizer = torch.optim.Adam(model.parameters(), 1e-5,
                                weight_decay=1e-4)

criterion = torch.nn.MSELoss()

test_accuracy = []
train_acc = []
for epoch in range(EPOCHS):
    tr, te = train(model, epoch, features_loader, orientation_loader, optimizer, criterion,
        valid_features_loader, valid_orientation_loader, LOG_INTERVAL, BATCH_SIZE)
    test_accuracy += te
    train_acc += tr
    
pickle.dump((test_accuracy,train_acc), open("/home/abenjamin/DNN_illusions/data/models/{}.traintest".format(savename),"wb"))
torch.save(model.state_dict(), "/home/abenjamin/DNN_illusions/data/models/{}.pt".format(savename))
print("Saved")