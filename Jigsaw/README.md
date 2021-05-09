# Learning features extraction through solving jigsaw puzzle
This folder contains the code and instructions to train jigsaw puzzle pretext task to learn to  extract features, and using it on image captioning downstream task.

# Instructions on running the code
## Getting started 
Create a conda environment using the dependencies file:

```sh
conda env create -f environment.yml
```
## Jigsaw pretext task
1. Create the HDF5 dataset (the model expects the data to be in HDF5 file) using the to_hdf5.py specifing the path and desired image size.
2. Create the desired number permutations using maximal_hamming.py in hamming_set by providing number of permutations, which will create a text file with the permutations.
3. In the main.py add the dataset path and the hamming_set (number of permutations) text file path and specify the number of permutations (max_hamming_set) to use from the permutations list (choose the same number if you want to use all the created permutations).
4. In case of using different image size, need to be specified in the dataset creation, datagenerator creation(image_size parameter) and in image_transform ( where crop size is the size of the random crop, cell size  is the size of large patch, and tile size is the final size of the patch).

## Downstream task
1. Using Jigsaw_feature_extraction.py, load the trained model and spacify dataset directory, and choose one of two functions  to extract the features: 
  - Full network uses the last dense layer before the soft max of the whole architecture for features extraction, must specify the same imgae size parameters used in training.
  -  Single network intialize a ResNet50 with the trained weights and use the GAP layer for feature extraction (not recommended).
2. Using jigsaw_vocabulary to generate the vocabulary which generates the descriptions.txt.
2. Using Jigsaw__IC_model to train the captioning model on the extracted features by spicifying the extracted features file location.
3. Jigsaw_test_blue.py to check the model blue score, requires only the extracted features file location.
4. jigsaw_test_images.py used to test the captioning models on images, need specify image location and the image caption model.

# pre-trained models
1. Pre-trained model for jigsaw with Resnet 50 with 67% accuracy on pretext task [Jigsaw_ResNet50](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/20020053_mbzuai_ac_ae/Ed2xPGXaqqpNuQfawHm5HvYBUbW4fL3HNLnTr9HAcrtDvQ?e=3OnR8N)
2. Pre-trained model of image captioning using the jigsaw extracted features [Image_captioning_jigsaw](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/20020053_mbzuai_ac_ae/EXHOb314z-1JlFZKr-umQ8kBOl_A9Q6s3ijJxWxknnheNQ?e=2Q00iC)
