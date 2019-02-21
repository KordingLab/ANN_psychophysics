import numpy as np
import cv2
import os
import pandas as pd

def png_to_array(image_path):
    image = cv2.imread(image_path, 0) #0 second argument means read as grayscale . . 1 = read as color, -1= read unchanged
    return image

def batch_folder_images(folder_path):
    directory = os.fsencode(folder_path)
    all_illusions = list()

    for file in os.listdir(directory):
        if file.endswith(".png"):
            image_path_string = os.path.join(directory, file)
            print(image_path_string)
            array_img = png_to_array(image_path_string)
            all_illusions.append(array_img.reshape(-1))
            continue
        else:
            continue

    return all_illusions


def save_h5(array):
    all_inputs = pd.DataFrame(np.stack(array))
    all_inputs.to_hdf('/home/rguan/data/illusions_h5/test1.h5', key="l", mode='w')


def main():
    print("starting")
    batch_array = batch_folder_images('/home/rguan/DNN_illusions/data/illusions_png/')
    save_h5(batch_array)

main()

