import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


# find


def compute_all_horizontal_slices(image_array, width):
    print(image_array.shape)
    row_length, column_length, channel_num = image_array.shape

    for r1 in range(row_length - width):
        r3 = r1 + width
        visualize_slice(image_array, r1, 0, r1, column_length - 1, r3, 0, r3, column_length - 1)


def visualize_slice(image_array, r1, c1, r2, c2, r3, c3, r4, c4):
    #     print("image_slice values")
    #     print(r1, r1 - (r1 - r3), c1, c1 - (c1 - c2))
    #     print("converted patch values")
    #     print(r3 + r1 - r3 ,r3, c3, c3 + c4 - c3)
    image_slice = image_array[r3 + r1 - r3: r3, c3: c3 + c4 - c3]

    #     print(image_slice)

    plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(131)
    plt.imshow(image_array)
    ax1.set_axis_off()
    ax1.set_title("Original")

    ax2 = plt.subplot(132)
    ax2.imshow(image_array)
    rect = patches.Rectangle((c3, r3), c4 - c3, r1 - r3, linewidth=1, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)
    ax2.set_axis_off()
    ax2.set_title("Slice ID-ed")

    ax3 = plt.subplot(133)
    ax3.imshow(image_slice)
    ax3.set_axis_off()
    ax3.set_title("Target Slice")

    plt.tight_layout()
    plt.show()


im = np.array(Image.open('/home/rguan/DNN_illusions/data/output/only_decoded_img_2.png'), dtype=np.uint8)
compute_all_horizontal_slices(im, 10)


