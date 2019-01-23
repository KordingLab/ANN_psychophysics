### Import things
import torch
from decoder import VGG_chopped
from data_generators import batch_inputs, get_quadratures

### define methods in torch to create the input data
KERNEL_SIZE = 15
filts = get_quadratures(KERNEL_SIZE)


### set which layer we'll decode from



### define the network: pull from a conv2d layer of the pretrained vgg network and train on top of that

vgg_chopped = VGG_chopped(5)

### on an long while loop, we create inputs and use them to train the top of the network

while 1:
    inputs, targets  = batch_inputs(filts)

    intermediates = vgg_chopped(inputs)


# params_to_update = model_ft.parameters()
# print("Params to learn:")
# if feature_extract:
#     params_to_update = []
#     for name,param in model_ft.named_parameters():
#         if param.requires_grad == True:
#             params_to_update.append(param)
#             print("\t",name)
# else:
#     for name,param in model_ft.named_parameters():
#         if param.requires_grad == True:
#             print("\t",name)

# save train performance to a csv



### finally we save the network for later