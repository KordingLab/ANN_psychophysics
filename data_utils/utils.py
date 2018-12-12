import torch
import pandas as pd

def features_dataset(features_pickle_path, train=True):
    """
    Loads the precomputed features.
    Returns a pytorch Dataset
    """
    if train:
        start = 0
        stop = 48000
    else:
        start = 48000
        stop = 50000

    features = pd.read_hdf(features_pickle_path, start=start, stop=stop).values

    labels = torch.zeros(features.shape[0]).float()
    features = torch.from_numpy(features).float()

    return torch.utils.data.TensorDataset(features, labels)

def orientations_generator(h5_path, batch_size, as_tensor=True, train=True):
    if train:
        start = 0
        stop = 48000
    else:
        start = 48000
        stop = 50000

    curr_index = start
    while 1:

        dataframe = pd.read_hdf(h5_path, start=curr_index,
                                stop=min([curr_index + batch_size, stop]))
        #         print("Indexes {} to {}".format(curr_index,curr_index+batch_size))
        curr_index += batch_size

        if curr_index >= stop:
            curr_index = start
            continue

        if as_tensor:
            out = torch.Tensor(dataframe.values).view(-1, 2, 224, 224)
        else:
            out = dataframe.values.reshape((-1, 2, 224, 224))
        yield out

def orientations_iterator(h5_path, batch_size, as_tensor=True, train=True):
    return iter(orientations_generator(h5_path, batch_size, as_tensor, train))