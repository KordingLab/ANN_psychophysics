# Shared visual illusions between humans and artificial neural networks

This repository is for sharing code for the ongoing collaboration with the[Stocker lab](https://www.sas.upenn.edu/~astocker/lab/members-files/alan.php).

### Organization

**Local_orientation_of_cross_lines**: (Ari and Ryan) Scripts for 
calculating the Fisher information with respect to the relative orientation
of crossed lines, as well as seeing if crossed line images lead to biases
in decoding (i.e. scripts for generating images, decoding local orientation 
 from the VGG16 network on these images, and testing illusions)
 
**evolving_Fisher**: (Ari) Why does the Fisher information look the way it does
in DNNs? Here the hypothesis is that it arises from the dynamics of gradient
descent.

**autogen**: (Ari and Ryan) Can we automatically generate new illusions?

**attic**: Archive of old scripts no longer in use.

### Submitted abstracts
#### CCN (poster)
Any information processing system should allocate resources where it matters: 
it should process frequent variable values with higher accuracy than less frequent ones. 
While this strategy minimizes average error, it also introduces an estimation bias. 
For example, human subjects perceive local visual orientation with a bias away from the orientations that occur most frequently in the natural world.
Here, using an information theoretic measure, we show that pretrained neural networks, 
like humans, have internal representations that overrepresent frequent variable values at the expense of certainty for 
less common values. Furthermore, we demonstrate that optimized readouts of local visual orientation from these networks’
internal representations show similar orientation biases and geometric illusions as human subjects. 
This surprising similarity illustrates that when performing the same perceptual task, similar characteristic illusions and biases emerge for any optimal information processing system that is resource limited.

##### DeepMath (rejected) (why this Fisher information develops in DNNs):

The inner representations of both deep learning systems and brains must contain information about important variables, like edge orientation, for which some values are more frequent than others. Brains are known to represent more frequent values with higher certainty than it represents less frequent values, and this leads to certain well-known visual illusions. This efficient representation scheme implies a scarcity of resources with which to minimize uncertainty, and is due to biological constraints like synaptic noise and spiking costs. Here we show that networks trained on ImageNet also learn representations in their early layers that have higher certainty for more frequent values of variables. We argue that this originates not from internal noise, as hypothesized for brains, but rather from the dynamics of gradient descent in multilayer networks. In deep linear networks, gradient descent is known to learn informative components in order of their importance. We show that uncertainty in hidden representations follows these transitions; the representation gains certainty about more informative components first. Thus, this similarity between human and deep neural network representations and the illusions they both exhibit arises from the dynamics of gradient descent. 