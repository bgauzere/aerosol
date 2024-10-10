# Contents

In this folder, you will find the following notebooks : 
 * `1_Segmentation_Watershed.ipynb`: Code to run a segmentation algorithm without learning a model
 * `2_Segmentation_Keras.ipynb` : Code to implement a segmentation model and learning it by yourself 
 * `3_Segmentation_SAM.ipynb` : Code to run the fundation model SAM to segment images.
 
 Their contents are described [below](#tutorials).

# Installation

## Prerequisites
To run the codes, you need : 
 * A set of images to test in ./Images/
 * the weights of SAM model :
   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth 
   Copy the file in the `assets` folder.

The Pipfile is made for CPU usage. 

## Get the EMPS dataset 

To get the EMPS dataset, clone the repository into `emps` folder : 
`git clone https://github.com/by256/emps.git`

## Jupyter notebook configuration 

1. Install pipenv : `pip install pipenv`
1. Install the venv : `pipenv install` 
1. Install Segment Anything : `pipenv run python -m pip install   git+https://github.com/facebookresearch/segment-anything.git`
1. Install the kernel : `pipenv run python -m ipykernel install --name="aerosol"   --userame="aerosol" --user`
1. Launch notebooks with `aerosol` jupyter kernel  


# Tutorials

If your local python configuration fails, you can run the [notebooks on colab](https://drive.google.com/drive/folders/1IbSBKymrI71-HVHzs5j9NkkAQ6qOmiB4?usp=sharing)

## `Segmentation_Watershed.ipynb`

## `Segmentation_Keras.ipynb`

version colab : 

Code pour faire un modèle simple de segmentation. Inspiré et adapté de de
F. Chollet : Deep Learning in Python.

## `Segmentation_Anything.ipynb` 
Code pour faire tourner un modele fondation sur des images quelconques. 


