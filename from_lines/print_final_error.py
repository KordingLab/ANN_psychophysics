import numpy as np
import pickle

for run in ["Linear_decoder_plain_lines_layer_", "Linear_decoder_upsample_plain_lines_layer_",
            "NonLinear_decoder_plain_lines_layer_"]:
    for layer in [ 4, 9, 16, 23, 30]:

        print("\n"+run+str(layer))
        train_accuracy = pickle.load( open("/home/abenjamin/DNN_illusions/data/models/{}{}.traintest".format(
                                run, layer), "rb"))
        print(np.mean(train_accuracy[-10:]))



