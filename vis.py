import matplotlib.pyplot as plt

def show(im):
    plt.imshow(im.detach().cpu(), cmap="gray")
    plt.xticks([])
    plt.yticks([])
    #plt.colorbar()
    plt.show()

import itk
def check(A, B):
    plt.imshow(itk.checker_board_image_filter(
        A.detach().cpu().numpy(),
        B.detach().cpu().numpy(),
        checker_pattern=9
    ))