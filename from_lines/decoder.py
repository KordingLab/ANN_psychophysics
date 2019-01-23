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

        # freeze to not retrain
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
       #  [batch_size, 64, 112, 112]

        return x


class OrientationDecoder(torch.nn.Module):
    """This class takes the inputs of the pretrained VGG function
    and runs it through deconvoltion to get the orientations back"""
    def __init__(self, layer):
        super(OrientationDecoder, self).__init__()
        self.layer = layer
        maxpool_indices = [ 4, 9, 16, 23, 30]
        assert layer in maxpool_indices

        # load the pretrained network
        self.vgg_chopped = VGG_chopped(layer)

        if self.layer == 4:
            # starts [batch_size, 64, 112, 112]
            self.deconv = torch.nn.Sequential(

            )

        elif self.layer == 9:
            # starts [64, 128, 56, 56]
            self.deconv = torch.nn.Sequential(

            )
        elif self.layer == 16:
            # starts [64, 256, 28, 28]
            self.deconv = torch.nn.Sequential(

            )
        elif self.layer == 23:
            # starts [64, 512, 14, 14]
            self.deconv = torch.nn.Sequential(

            )
        elif self.layer == 30:
            # starts [64, 512, 7, 7]
            self.deconv = torch.nn.Sequential(

            )
        else:
            NotImplementedError("Impossible logic")

    def forward(self, x):
        x = self.vgg_chopped(x)
        x = self.deconv(x)



        assert x.size()[1:] == torch.Size([2, 224,224])

        return x