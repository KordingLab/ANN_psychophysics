import torch
import torch.nn as nn


class Decoder(torch.nn.Module):
    """This network decodes the features of a layer of the VGG network into an orientation image.
    The orientation image has size 224x224 and two channels, which together define the orientation vector
    at each pixel."""
    def __init__(self, num_ftrs):
        super(Decoder, self).__init__()

        if num_ftrs == 512 * 7 * 7:
            # this means we're using the maxpool 31 layer
            self.layer = "31"
        elif num_ftrs == 512 * 14 * 14:
            # this means we're using the maxpool 24 layer
            self.layer = "24"
        else:
            raise NotImplementedError("Need to set up the decoder for this layer")

        if self.layer == "31":
            self.first_deconv = torch.nn.Sequential(

                torch.nn.ConvTranspose2d(512, 64, 6, 4, 1),
                nn.BatchNorm2d(64),
                nn.ReLU())

        elif self.layer == "24":
            self.first_deconv = torch.nn.Sequential(

                torch.nn.ConvTranspose2d(512, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU())

        self.deconv = torch.nn.Sequential(

            # 64 x 28 * 28
            torch.nn.ConvTranspose2d(64, 16, 4, 2, 1),
            #             nn.BatchNorm2d(16),
            nn.ReLU(),
            # 16 x 56 x 56
            torch.nn.ConvTranspose2d(16, 2, 6, 4, 1),
            # 2 x 224 x 224
        )

    def forward(self, x):
        if self.layer == "31":
            x = x.view(-1, 512, 7, 7)
        elif self.layer == "24":
            x = x.view(-1, 512, 14, 14)

        x = self.first_deconv(x)
        x = self.deconv(x)

        return x