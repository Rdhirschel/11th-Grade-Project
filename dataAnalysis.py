import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import numpy as np
import seaborn as sns

# check how many images are in each folder
def check_image_count(image_directory="dataset\\train"):
    min = 1000000000
    sum = 0
    #classNames = ["tulip", "sunflower", "rose", "dandelion", "iris"]
    classNames = os.listdir(image_directory)
    for folder_name in classNames:
        count = 0
        for file_name in os.listdir(os.path.join(image_directory, folder_name)):
            count += 1
            if (count == 1):
                image = Image.open(os.path.join(image_directory, folder_name, file_name))
                image.show()
        if count < min:
            min = count
        print(f"Folder {folder_name} has {count} images") # for equal data, each folder should have the same number of images
        sum += count

    print (f"Minimum number of images in a folder is {min}")
    print (f"Total number of images is {sum}")

check_image_count()