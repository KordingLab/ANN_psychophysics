import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torchvision
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import h5py
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F

import torchvision.models as models



class VGG_l6(torch.nn.Module):

    def __init__(self, model, layer):
        super(VGG_l6, self).__init__()
        self.features = model.features
        self.classifier = list(model.classifier.children())[0]
        self.layer = layer

        self.maxpool_indices = [0, 5, 10, 17, 24, 31]

    def forward(self, x):
        for i in range(len(self.maxpool_indices) - 1):
            x = self.features[self.maxpool_indices[i]:self.maxpool_indices[i + 1]](x)
            if i == self.layer:
                return x
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




if __name__ == "__main__":
    valdir = '/home/abenjamin/DNN_illusions/data/illusions'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(150),
            transforms.CenterCrop(224),
            #         transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset_nonnormalized = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(150),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]))

    val_loader = torch.utils.data.DataLoader(val_dataset_nonnormalized,
                                             batch_size=1, shuffle=False,
                                             num_workers=1, pin_memory=True)

    i = 0
    for image, label in val_dataset_nonnormalized:
        image = np.squeeze((np.moveaxis(image.numpy(), 0, -1)))
        plt.subplot(121)
        fig = plt.imshow(image)
        plt.axis('off')
        plt.show()

    # resnet18 = models.resnet18(pretrained=True)
    # alexnet = models.alexnet(pretrained=True)
    # squeezenet = models.squeezenet1_0(pretrained=True)
    model = models.vgg16(pretrained=True)
    # densenet = models.densenet161(pretrained=True)
    # inception = models.inception_v3(pretrained=True)

    n_illusions = bs = 3

    features = ['vgg_maxpool5_illusions',
                'vgg_maxpool10_illusions',
                'vgg_maxpool17_illusions',
                'vgg_maxpool24_illusions',
                'vgg_maxpool31_illusions',
                'vgg_fc1_illusions']

    # list of hdf stores
    all_features = list()

    # I've set this up so that it could save all the features each run. However for 50000 images this is too large
    # for memory. Therefore I run a loop, and save a feature a iteration. Note that this requires running the network
    # on the same image multiple times
    for feature_number in range(len(features)):
        #     if feature_number <5:
        #         continue
        print(features[feature_number])

        vgg_features = VGG_l6(model, feature_number).cuda()

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=bs, shuffle=False,
                                                 num_workers=4, pin_memory=True)

        # switch to evaluate mode
        vgg_features.eval()

        for i, (input_var, _) in tqdm(enumerate(val_loader)):
            with torch.no_grad():
                input_var = torch.autograd.Variable(input_var).cuda()

            # compute features, one per layer
            feature_list = vgg_features(input_var)

            # add this particular feature to that list
            all_features.append(feature_list.detach().cpu().numpy())

        #         if (i%(len(val_loader)//10) == 0 ) and i>0:
        all_features = pd.DataFrame(np.vstack(all_features).reshape(n_illusions, -1))
        all_features.to_hdf('data/features/Illusions/{}.h5'.format(features[feature_number]),
                            key='{}'.format(features[feature_number]), mode='w')
        del all_features
        all_features = list()

    feature_list.size()




