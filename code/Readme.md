# Contents

In this folder, you will find the following notebooks : 
 * `1_Segmentation_Watershed.ipynb`: Code to run a segmentation algorithm without learning a model
 * `2_Segmentation_Keras.ipynb` : Code to implement a segmentation model and learning it by yourself 
 * `3_Segmentation_SAM.ipynb` : Code to run the fundation model SAM to segment images.

This folder contains the uncomplete notebooks. Complete versions are in the `solution`folder.
 
Notebook contents are described [below](#tutorials).

# Installation

## Prerequisites

To run the codes, you need : 
 * A set of images to test in ./images_raw/
 * the weights of SAM model :
   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth 
   Copy the file in the `assets` folder.
  * The EMPS dataset : To get the EMPS dataset, clone the repository into `emps` folder : 
`git clone https://github.com/by256/emps.git` or download the zip file from the repository https://github.com/by256/emps and extract it in the `emps` folder.


The Pipfile is made for CPU usage. 

## Python configuration 

To run the notebooks, we propose two alternatives, depending on your preferences. Setting up the environment may take some time depending on your particular configuration.

### Conda/Anaconda

To use these notebooks with conda, launch the following commands within the anaconda prompt :
1. create a new environment with the following command : 
`conda env create --file aerosol_ia.yml`

2. Then activate the environment : 
`conda activate aerosol_ia`

3. And launch jupyter :
`jupyter lab`

### Pip + Pipenv

If you prefer to use pip and pipenv, you can follow these steps :

1. Install pipenv : `pip install pipenv`
1. Install the venv : `pipenv install` 
1. Install Segment Anything : `pipenv run python -m pip install   git+https://github.com/facebookresearch/segment-anything.git`
1. Install the kernel : `pipenv run python -m ipykernel install --name="aerosol"   --userame="aerosol" --user`
1. Launch notebooks with `aerosol` jupyter kernel  


# Tutorials

If your local python configuration fails, you can run the [notebooks on colab](https://drive.google.com/drive/folders/1IbSBKymrI71-HVHzs5j9NkkAQ6qOmiB4?usp=sharing)

## Segmentation_Watershed.ipynb
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11lgfZxdbgAsnCZ18Oaj-iduRAU7rTcr4?usp=sharing)

This notebook provides an in-depth exploration of image segmentation using the Watershed algorithm. The notebook covers key concepts of the watershed technique, explains the preprocessing steps required for efficient segmentation, and includes practical examples of applying the algorithm to different types of images.

The following topics are covered:
  - Image preprocessing techniques (e.g., grayscale conversion, thresholding, and noise removal)
  - Marker-based Watershed Segmentation
  - Visualizations to illustrate intermediate and final results

The examples are implemented using OpenCV, NumPy, and Matplotlib, and they demonstrate the capabilities of the Watershed algorithm to segment images.


## Segmentation_Keras.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fL-T8icnZgEHnBakr_ClB1yyQR6_q-Lp?usp=sharing)

This notebook presents a step-by-step guide to image segmentation using a deep learning model built with Keras. It provides a practical implementation of segmentation techniques commonly used in medical imaging, satellite imagery, or general computer vision applications.

The notebook includes:

  - An introduction to image segmentation and its applications.
  -  Loading and preprocessing of image datasets for training and validation.
  -  Building a convolutional neural network (CNN) using the Keras framework, specifically tailored for segmentation tasks.
  -  Explanation of model architecture, including commonly used layers such as convolutional, pooling, and upsampling.
  -  Training the segmentation model, including details on loss functions  and metrics for evaluation.
  -  Visualizing the model's performance with segmented output images.

This code has been inspired by and adapted from the book "Deep Learning with Python" by François Chollet, the creator of Keras.

## Segmentation_Anything.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19Bce8yQdlOCjFMSd98DKklddNNoclwU5?usp=sharing)


This notebook explores the Segment Anything Model (SAM) for image segmentation. SAM is a versatile and efficient model designed to perform general-purpose segmentation on a wide range of images with minimal user input. This notebook demonstrates how to leverage SAM for different image segmentation tasks.

Key components of the notebook include:

  - Introduction to the Segment Anything Model (SAM) and its capabilities.
  - Loading images and preparing them for segmentation.
  -  Utilizing pre-trained SAM for segmenting various objects within images.
  -  Visualization of segmentation masks produced by the model.
  

