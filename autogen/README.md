# Automated generation of novel illusions

This package leverages a new theory of visual illusions to generate new illusions of new types.

In human vision, there is a robust connection between perceptial bias (i.e. an illusion) and discrimination errors (i.e. uncertainty).
(see [this paper](https://www.sas.upenn.edu/~astocker/lab/publications-files/journals/PNAS2017/Wei_Stocker2017.pdf)or
[this paper too](https://www.sas.upenn.edu/~astocker/lab/publications-files/journals/NN2015/Wei_Stocker2015b.pdf)).


Since pretrained deep neural networks show similar uncertainty about the world (in a Fisher information sense), 
we can use them as proxies for human uncertainty. This allows us to find new illusions in an automated fashion.


#### Algorithm
To create an image with an illusory percept for thing x (e.g. curvature of some line, lightness, etc.)

1) Define a parameterized image generator G with x as a smoothly-varying input parameter. 
2) For any generated image, estimate to Fisher of a pretrained DNN w/r/t x
3) Sample from the density of G parameters that yield images for which the Fisher for x changes quickly, using deep density techniques (e.g. GANs)


#### Organization

In `image_generators.py`, we define our image generators. Each subclass of `BaseImageGenerator` has methods to generate
an image deterministically and smoothly from "context" inputs and a 1D "theta" input.

In `fisher_estimation.py`, we define scripts that calculates the Fisher information of a pretrained deep neural network
layer as a function of "theta", given some "context".

In `density_estimation.py`, we define scripts that can sample from the density of "context" variables to find values 
where the Fisher information w/r/t theta has a large value. 



