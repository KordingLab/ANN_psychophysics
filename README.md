# DNN_illusions
Do deep neural networks have visual illusions?


### To run on bleen

0. Make sure you're in the `py34` environment on bleen. (Get there with `source activate py34`). 
0. Select an open gpu. (Run `nvidia-smi`, see which cards are unused, and change the `CUDA_VISIBLE_DEVICES` flag in the top of `train-decoder.py` to match).

### Performance log


| Date      | Orientation Kernel | Layer number | Optimizer     | Other notes | Performance |
|-----------|--------------------|--------------|---------------|-------------|-------------|
| 12-15-18  | 30                 | 24           | Adam: lr=1e-5 |             | 0.020909    |
|           |                    |              |               |             |             |
|           |                    |              |               |             |             |
