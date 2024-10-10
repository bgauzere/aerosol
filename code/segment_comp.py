# Description: This file contains the code for the segmentation comparison
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import keras
import segmentation_models as sm
import matplotlib.pyplot as plt
import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


# ----------------- Utils -------------------------------- #
def show_image(ax, folder_path, image_name):
    image_path = os.path.join(folder_path, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Convert image to RGB if it is in grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return image


# ----------------- Segmentation methods ----------------- #
def seg_watershed(ax, image):
    """Segmentation using the watershed algorithm
    (https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html)
    Non-ML method, used here as a baseline.

    :param ax: a PyPlot Axes object in which to plot
    :param image: the image to segment (as a numpy array as obtained by cv2.imread)
    :return: the segmented image
    """
    # important thresholding param for detecting what is sure foreground
    # depends on the size of what you want to segment and on the grayscale histogram
    lam = 0.005
    # works on a copy because it prints the segmentation contours in the image
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, lam * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    ax.imshow(img)
    ax.axis('off')
    return img


def seg_sam(ax, image):
    """Segmentation using the SAM model
    (https://github.com/facebookresearch/segment-anything)
    Generate the segmentation masks of a given image with the SAM model, and display the contours on the image.

    :param ax: a PyPlot Axes object in which to plot
    :param image: the image to segment (as a numpy array as obtained by cv2.imread)
    :return: the segmented image
    """
    sam = sam_model_registry["default"](checkpoint="./assets/sam_vit_h_4b8939.pth")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    if len(masks) == 0:
        return
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    img = np.ones((sorted_masks[0]['segmentation'].shape[0], sorted_masks[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for mask in sorted_masks:
        m = mask['segmentation']
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0, 0.4), 3)  # Draw contours in green
    ax.imshow(image)
    ax.imshow(img)
    ax.axis('off')
    return img


def seg_unet(ax, image):
    """Segmentation using the U-Net model from the segmentation_models library
    Work in progress...
    """
    model = sm.Unet('resnet34', encoder_weights='imagenet')


# ----------------- Main --------------------------------- #
if __name__ == '__main__':
    folder_name = './Images/CEA2_MEB'
    image_name = "Image_4_01.tif"
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    image = show_image(ax, folder_name, image_name)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    img_w = seg_watershed(axes[0], image)
    img_sam = seg_sam(axes[1], image)
    plt.show()



