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


class AngleDecoder(torch.nn.Module):
    """This class takes the inputs of the pretrained VGG function and gives out the angle
    """
    def __init__(self, layer, noise = 0, nonlinear = False):
        super(OrientationDecoder, self).__init__()
        self.layer = layer
        self.noise = noise
        maxpool_indices = [ 4, 9, 16, 23, 30]
        assert layer in maxpool_indices

        # load the pretrained network
        self.vgg_chopped = VGG_chopped(layer)

        n_feats = {4: 64 * 112 * 112,
                   9: 128 * 56 * 56,
                  16: 256 * 28 * 28,
                  23: 512 * 14 * 14,
                  30: 512 * 7 * 7}

        if nonlinear:
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(n_feats[layer], 50),
                torch.nn.Dropout(.5),
                torch.nn.Linear(50, 1)
            )
        else:
            self.decoder = torch.nn.Sequential(
                        torch.nn.Linear(n_feats[layer],1)
                        )


        self.n_feat = n_feats[layer]


    def forward(self, x):
        x = self.vgg_chopped(x)
        # flatten
        x = x.view(-1,self.n_feat)
        # add noise
        x += self.noise * torch.randn(*x.size())
        # get angle
        x = self.deconv(x)

        return x
