# Efficient neural codes naturally emerge through gradient descent learning

This repository hosts code for "Efficient neural codes naturally emerge through gradient descent learning" available at [https://www.nature.com/articles/s41467-022-35659-7](https://doi.org/10.1038/s41467-022-35659-7)

This is a collaboration with the [Stocker lab](https://www.sas.upenn.edu/~astocker/lab/members-files/alan.php).

### Local setup
**conda env create -f environment.yml**

### Organization

All subpanels for figures in the paper can be produced by running code in the appropriate notebook in `figure_notebooks`. Many figures can be run locally, but some will strictly require cuda. All notebbooks can be run on Google Colab. 

### External files 

Imagenet crops can be downloaded at: https://drive.usercontent.google.com/download?id=1mF46SUDKzG0LkWkNGV1fP2hTEgW5WbF\_

|   | Run | View |
| - | --- | ---- |
| Figure 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quietscientist/ANN_psychophysics/blob/master/figure_notebooks/Fig_2_orientation.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/quietscientist/ANN_psychophysics/blob/master/figure_notebooks/Fig_2_orientation.ipynb?flush_cache=true) |
| Figure 3| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](hhttps://colab.research.google.com/github/quietscientist/ANN_psychophysics/blob/master/figure_notebooks/Fig_3_hue_sensitivity.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/quietscientist/ANN_psychophysics/blob/master/figure_notebooks/Fig_3_hue_sensitivity.ipynb?flush_cache=true) |
| Figure 4| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quietscientist/ANN_psychophysics/blob/master/figure_notebooks/Fig_4_Linear_demo.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/quietscientist/ANN_psychophysics/blob/master/figure_notebooks/Fig_4_Linear_demo.ipynb?flush_cache=true) |
| Figure 6| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quietscientist/ANN_psychophysics/blob/master/figure_notebooks/Fig_6_supervised_demo.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/quietscientist/ANN_psychophysics/blob/master/figure_notebooks/Fig_6_supervised_demo.ipynb?flush_cache=true) |
| Train on Rotated CIFAR| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quietscientist/ANN_psychophysics/blob/master/scripts/train_rotated.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/quietscientist/ANN_psychophysics/blob/master/scripts/train_rotated.ipynb?flush_cache=true) |



