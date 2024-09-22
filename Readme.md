# Contents
Code pour faire tourner un modele fondation sur des images quelconques

# Installation

To run the code, you need : 
 * A set of images in ./Images/
 * the weights of SAM model :
   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth 


The Pipfile is made for CPU usage. 

# Run 

1. Install the venv : `pipenv install` 
1. Install Segment Anything : `pipenv run python -m pip install   git+https://github.com/facebookresearch/segment-anything.git`
1. Install the kernel : `pipenv run python -m ipykernel install --name="aerosol"   --userame="aerosol" --user`
 1. Launch notebooks with `aerosol` jupyter kernel  

# Tests
 * Segmentation models : limited to 21 channel image, no clear regions. 
 * Segment Anything : long to compute, but good results. (1m 30 on my CPU for 1 image)
 * https://stardist.net/ : TODO
 * weka : https://github.com/fiji/Trainable_Segmentation : TODO
 * https://github.com/MouseLand/cellpose  : TODO
