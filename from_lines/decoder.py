import torchvision.models as models
import torch


#### Import the pretrained model ####



class VGG_chopped(torch.nn.Module):
    """This class cuts the pretrained VGG function at a layer and outputs the activations there."""
    def __init__(self, layer):
        super(VGG_chopped, self).__init__()
        features = list(models.vgg16(pretrained = True).features)[:layer+1]
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


        # see https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md and
        # https://distill.pub/2016/deconv-checkerboard/ to think about the deconv

        # Currently a single linear layer
        if self.layer == 4:
            # starts [batch_size, 64, 112, 112]
            self.deconv = torch.nn.Sequential(
                  #  112 x stride - 2 x padding + kernel + output_pad
                torch.nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=4, stride=2, padding=2),
            )

        elif self.layer == 9:
            # starts [64, 128, 56, 56]
            self.deconv = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=128, out_channels=2, kernel_size=4, stride=4, padding=2),
            )
        elif self.layer == 16:
            # starts [64, 256, 28, 28]
            self.deconv = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=256, out_channels=2, kernel_size=8, stride=8, padding=4),
            )
        elif self.layer == 23:
            # starts [64, 512, 14, 14]
            self.deconv = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=256, out_channels=2, kernel_size=16, stride=16, padding=8),
            )
        elif self.layer == 30:
            # starts [64, 512, 7, 7]
            self.deconv = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=512, out_channels=2, kernel_size=32, stride=32, padding=16)
            )
        else:
            NotImplementedError("Impossible logic")

    def forward(self, x):
        x = self.vgg_chopped(x)
        x = self.deconv(x)

        assert x.size()[1:] == torch.Size([2, 224,224])

        return x