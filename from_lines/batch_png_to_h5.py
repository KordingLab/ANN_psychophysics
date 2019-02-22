import numpy as np
import cv2
import os
import pandas as pd

def png_to_array(image_path):
    string_path = image_path.decode('ascii')
    print("IMAGE PATH: ", string_path)
    image = cv2.imread(string_path, 0) #0 second argument means read as grayscale . . 1 = read as color, -1= read unchanged
    three_channel_image = np.zeros((3, 224, 224))

    three_channel_image[0] = image[:,:] * (1.0/255) #lazy and perhaps inefficient way of copying arrays. . . Might change later
    three_channel_image[1] = image[:,:]  * (1.0/255)
    three_channel_image[2] = image[:,:] * (1.0/255)
    print("3 channel shape: ", three_channel_image.shape)
    return three_channel_image

def batch_folder_images(folder_path):
    directory = os.fsencode(folder_path)
    all_illusions = list()

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
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
    all_inputs.to_hdf('/home/rguan/DNN_illusions/data/illusions_h5/test1.h5', key="l", mode='w')


def main():
    print("starting")
    batch_array = batch_folder_images('/home/rguan/DNN_illusions/data/illusions_png/')
    save_h5(batch_array)

main()

