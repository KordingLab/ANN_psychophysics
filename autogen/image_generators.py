import numpy as np
import torch

class BaseDifferentiableImageGenerator(object):

    """Base class for all image generators that are written in pytorch and differentiable into theta.
    All classes should output an image of ImageNet dimensions (224x224).
    """

    def __init__(self):

        pass

    def generate_image(self, theta, context):
        """To be redefined in any subclass.

        Deterministically generates an image as a function of theta and context.

        Context and theta should both smoothly relate to the image. Small changes in
        theta and changes in context should cause small changes in the output image.
        For two values of theta but the same context, the two output images should be
        the same in all manners except for theta.

        Methods need to be written in pytorch, such that they take a Variable and produce an
        image that is differentiable with respect to theta

        """
        raise NotImplementedError()


class BaseNonDifferentiableImageGenerator(object):
    """Base class for all image generators that aren't differentiable.

    All classes should output an image of ImageNet dimensions (224x224).

    When possible, all methods should still use torch methods and not numpy or scipy methods, for speed."""

    def __init__(self):
        pass

    def generate_image(self, theta, context):
        """To be redefined in any subclass.

        Deterministically generates an image as a function of theta and context.

        Context and theta should both smoothly relate to the image. Small changes in
        theta and changes in context should cause small changes in the output image.
        For two values of theta but the same context, the two output images should be
        the same in all manners except for theta.

        """
        raise NotImplementedError()

class OneCurvedLineGenerator(BaseDifferentiableImageGenerator):
    """This is designed to create illusions like the Herring illusion.
    """

    def __init__(self, n_lines = 5):
        super(OneCurvedLineGenerator, self).__init__()
        self.n_lines = n_lines


    def generate_image(self, theta, context):
        """
    Theta is the curvature of the central horizontal line.
    Context are parameters that describe the overlaid lines.
    e.g. a list of midpoints and orientations, for a total of n_lines x 3 parameters. """

        raise NotImplementedError()



class CentralPixelGenerator(BaseDifferentiableImageGenerator):
    """This is designed to generate images with a central block of pixels whose lightness is an illusions.
    """

    def __init__(self, n_pixels_blocks_per_side = 5):
        """


        :param n_pixels_blocks_per_side: an odd integer
        """
        super(CentralPixelGenerator, self).__init__()
        assert n_pixels_blocks_per_side % 2 == 1, "n_pixels_blocks_per_side must be odd"

        self.n_pixels_blocks_per_side = n_pixels_blocks_per_side


    def generate_image(self, theta, context):
        """
        :param theta: the lightness of the central pixel block.
        :param context: the lightness of the surrounding pixels.
        :return: A 224x224x3 image
        """
        assert len(context) == self.n_pixels_blocks_per_side**2-1

        raise NotImplementedError()