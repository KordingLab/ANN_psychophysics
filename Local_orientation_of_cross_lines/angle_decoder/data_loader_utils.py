import pandas as pd
import torch

def loader_generator(h5_path, batch_size, as_tensor=True, n_samples = 10000):
    """Given an h5 path to a file that holds the arrays, returns a generator
    that can get certain data at a time."""

    stop = n_samples
    curr_index = 0
    start = 0
    while 1:

        dataframe = pd.read_hdf(h5_path, start=curr_index,
                                stop=min([curr_index + batch_size, stop]))
        curr_index += batch_size

        if (dataframe.shape[0]==0) or (curr_index >= stop):
            curr_index = start
            continue

        if as_tensor:
            if dataframe.shape[1]>1:
                out = torch.Tensor(dataframe.values).view(batch_size, -1, 224, 224)
            else:
                out = torch.Tensor(dataframe.values)
        else:
            if dataframe.shape[1] > 1:
                out = dataframe.values.reshape((batch_size, -1, 224, 224))
            else:
                out = dataframe.values
        yield out

def data_iterator(h5_path, batch_size, as_tensor=True):
    return iter(loader_generator(h5_path, batch_size, as_tensor))