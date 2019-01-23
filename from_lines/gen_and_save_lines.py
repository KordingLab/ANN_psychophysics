from data_generators import create_input_and_target, get_quadratures
import pandas as pd
import numpy as np

KERNEL_SIZE = 15
filts = get_quadratures(KERNEL_SIZE)

# list of hdf stores
all_inputs = list()
all_targets = list()
n_samples = 10000
for n in range(n_samples):
    input, target  = create_input_and_target(filts)
    all_inputs.append(input.reshape(1,-1))
    all_targets.append(target.reshape(1,-1))


all_inputs = pd.DataFrame(np.stack(all_inputs))
all_inputs.to_hdf('data/features/lines.h5', mode='w')

all_targets = pd.DataFrame(np.stack(all_targets))
all_targets.to_hdf('data/features/lines_targets.h5', mode='w')



