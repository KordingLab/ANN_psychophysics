import torchvision.models as models
import torch


#### Import the pretrained model ####



class VGG_chopped(torch.nn.Module):
    """This class cuts the pretrained VGG function at a layer and outputs the activations there."""
    def __init__(self, layer):
        super(VGG_chopped, self).__init__()
        features = list(models.vgg16(pretrained = True).features)[:layer+1]
        print("Using layer {}:{}".format(layer,features))

        self.features = torch.nn.Sequential(*features).eval()
        self.layer = layer
        maxpool_indices = [ 5, 9, 16, 23, 30]
        assert layer in maxpool_indices

        # freeze for training
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        print(x.size())

        return x
