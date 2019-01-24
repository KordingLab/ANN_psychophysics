import pandas as pd
import torch

def loader_generator(h5_path, batch_size, as_tensor=True, n_samples = 10000):
    """Given an h5 path to a file that holds the arrays, returns a generator
    that can get certain data at a time."""

    stop = n_samples
    curr_index = start = 0
    while 1:

        dataframe = pd.read_hdf(h5_path, start=curr_index,
                                stop=min([curr_index + batch_size, stop]))
        curr_index += batch_size

        if curr_index >= stop:
            curr_index = start
            continue

        if as_tensor:
            out = torch.Tensor(dataframe.values).view(batch_size, -1, 224, 224)
        else:
            out = dataframe.values.reshape((batch_size, -1, 224, 224))
        yield out

def data_iterator(h5_path, batch_size, as_tensor=True):
    return iter(loader_generator(h5_path, batch_size, as_tensor))