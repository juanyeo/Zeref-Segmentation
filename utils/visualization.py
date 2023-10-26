from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

def show_image_seg(file_name, prediction, colormap='Reds', save_name='result'):
    plt.figure()
    ax = plt.gca()
    # show image
    image = Image.open(file_name).convert("RGB")
    ax.imshow(image)
    # show segmentations
    print(prediction.shape)
    segmentation = F.interpolate(prediction.reshape(1, 1, 352, 352), size=(image.height, image.width)).reshape(image.height, image.width)
    ax.imshow(segmentation, cmap=colormap, alpha=0.6)
    #plt.show()
    plt.savefig(save_name + '.png')