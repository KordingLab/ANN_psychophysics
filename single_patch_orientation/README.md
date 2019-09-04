# Single patch orientation using pre-trained AlexNet

Test pre-trained AlexNet with various orientation stimuli used in psychophysics to see if the model shows similar behaviors as in human observers.

## Usage
The main Python file for this project is `alexnet_to_orientation.py`.
An example usage of this script is:

```
 $ python alexnet_to_orientation.py --epochs 10 \
   --save-model --model-name 'alexNet_broadband_multiorinoise_naturaloriprior' \
   --if-more-noise-levels
```

For all options and command-line arguments, please use:

```
 $ python alexnet_to_orientation.py -h
```
