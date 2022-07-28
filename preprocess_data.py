import itk
import glob
import numpy as np
import matplotlib.pyplot as plt
import icon_registration as icon
import icon_registration.networks as networks
import torch
import random
import csv
import SimpleITK as sitk

def radiograph_dataset_s():
    return torch.load("/playpen/tgreer/radio_dataset")


def dxa_dataset():
    roots = glob.glob("dxa2d_processed/dxa2d/*")

    paths = [sorted(glob.glob(f"{root}/*.dcm")) for root in roots]

    knees = []
    for path in paths:
        for subpath in path:
            im = np.array(itk.imread(subpath))[0]
            if im.shape[1] == 640:
                knees.append(im)
    flipped_knees = []

    for im in knees[:84:2]:
        flipped_knees.append(im)
        #plt.imshow(im)
        #plt.show()

    for im in knees[85::2]:
        flipped_knees.append(im)
        #plt.imshow(im)
        #plt.show()

    for im in knees[1:85:2]:
        im = np.flip(im, axis=1)
        flipped_knees.append(im)
        #plt.imshow(im)
        #plt.show()

    for im in knees[86::2]:
        im = np.flip(im, axis=1)
        flipped_knees.append(im)
        #plt.imshow(im)
        #plt.show()
    big_enough_knees = []
    for im in flipped_knees:
        if im.shape[0] >= 845:
            big_enough_knees.append(im[:845])
    big_enough_knees = torch.tensor(np.array(big_enough_knees))[:, None, :, :] / 255.

    test_imgs = big_enough_knees[-10:]
    big_enough_knees = big_enough_knees[:-10]
    return big_enough_knees, test_imgs

def radiograph_dataset():
    r = csv.reader(open("./radiographs.csv"))
    radiograph_info = list(r)
    images = []
    for i in range(1000):
        reader = sitk.ImageSeriesReader()
        img_names = reader.GetGDCMSeriesFileNames(f"/playpen-raid/data/OAI/{radiograph_info[i][0]}")
        reader.SetFileNames(img_names)
        image = sitk.GetArrayFromImage(reader.Execute())[0]
        images.append(image)
    import matplotlib.pyplot as plt
    import numpy as np
    shapes = [i.shape for i in images]
    ratios = [2 * s[0] / s[1] for s in shapes]
    prepared_images = []
    for input_image in images:
        input_image = input_image[:, :input_image.shape[1] // 2]


        correct_height = input_image.shape[1] * 845 / 640


        shave = round((input_image.shape[0] - correct_height ) / 2)
        if shave >= 0:

            output_image = input_image[shave:-shave]

            output_image = torch.tensor(output_image / 6000.)
            output_image = torch.clip(output_image, 0, 1)[None, None]

            output_image = torch.nn.functional.interpolate(output_image, size=(845, 640), mode="bilinear")

            prepared_images.append(output_image)
    prepared_images = torch.cat(prepared_images)
    return prepared_images[:-100], prepared_images[-100:]