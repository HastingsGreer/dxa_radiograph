import torch
import matplotlib.pyplot as plt
import cv2

with open("categories.txt", "w") as f:
    ims = torch.load("radio_dataset")
    print(ims.shape)
    for im in ims:
        cv2.imshow("fuck", im[0].numpy() / torch.max(im[0]).numpy())
        val = cv2.waitKey()
        print(val)
        if val == 113:
            break
        f.write(str(val) + "\n")


