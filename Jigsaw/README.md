# Instructions on running the code
## pretext
1. create the HDF5 dataset (the model expects the data to be in HDF5 file) using to_hdf5.py specifing the path and desired image size.
2. create the permutations using maximal_hamming.py in hamming_set by providing number of permutations which will create a text file with the permutations
3. in the main.py add the dataset path and the hamming_set (number of permutations) text file path and specify the number of permutations (max_hamming_set) 
4. in case of using different image size, need to be specified in the dataset creation, datagenerator creation(image_size parameter) and in image_transform ( where
 crop size is the size of the random crop, cell size  is the size of large patch, and tile size is the final size of the patch)

## down-stream
1. using 1_Jigsaw_generator.py, load the trained model and spacify dataset directory, and choose one of two functions  to extract the features: 
  - Full network uses the last dense layer before the soft max of the whole architecture for features extraction.
  -  single network intialize a ResNet50 with the trained weights and use the GAP layer for feature extraction (not recommended).
2. using 2_Jigsaw_model.py to train the captioning model on the extracted features by spicifying the extracted features file location.
3. 3_Jigsaw_blue.py to check the model blue score, requires only the extracted features file location
4. 4_jigsaw_test.py used to test the captioning models on images, need specify image location and the image caption model.

# pre-trained models
1. pre-trained model for jigsaw with Resnet 50 with 67% accuracy on pretext task [Jigsaws_ResNet50](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/20020053_mbzuai_ac_ae/Ed2xPGXaqqpNuQfawHm5HvYBUbW4fL3HNLnTr9HAcrtDvQ?e=3OnR8N)
2. pre-trained model of image captioning using the jigsaw extracted features [Image captioning_jigsaw](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/20020053_mbzuai_ac_ae/EXHOb314z-1JlFZKr-umQ8kBOl_A9Q6s3ijJxWxknnheNQ?e=2Q00iC)
