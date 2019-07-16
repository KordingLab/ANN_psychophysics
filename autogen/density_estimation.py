import torch

""" These scripts aim to find values of 'context' that have highly sloped Fisher 
information in pretrained network representations. 

There are a few possibilities. All of them would benefit from speed, and smoothness of the map from context to Fisher.
In order of complexity:

1.  Optimization
----------------
For low-dimensional contexts, we could use an out-of-the-box, general purpose optimizer, e.g. in scipy. This approach
is more promising when the context is not expected to have image-like structure (see 'Deep Image Prior'). 

Initially, we'll simply optimize a neural network to transform random noise into the context parameters, with the loss
being the squared slope of the Fisher at that value of context. 
If degenerate modes are found, we might have to impose an entropy constraint on the output. 

2. Sampling
-----------
Using the derivative of the Fisher as the likelihood, or energy, we can also apply MCMC sampling methods. This is 
probably more likely to work in higher-dimensional situations.

If this shows slow convergence, we can attempt new MCMC methods that additionally incorporate a model that selects
which things are sampled. (e.g. https://papers.nips.cc/paper/7099-a-nice-mc-adversarial-training-for-mcmc.pdf)

"""