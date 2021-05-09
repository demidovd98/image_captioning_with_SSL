# Instructions on running the code
## pretext
1. create the HDF5 dataset which the model expect using to_hdf5.py specifing the path and desired image size
2. create the permutations using maximal_hamming.py in hamming_set by providing number of permutations which will create a text file with the permutations
3. in the main.py add the dataset path and the hamming_set (number of permutations) text file path and specify the number of permutations (max_hamming_set) 
4. in case of using different image size, need to be specified in the dataset creation, datagenerator creation(image_size parameter) and in image_transform ( where
5.  crop size is the size of the random crop, cell size 6. is the size of large patch, and tile size is the final size of the patch)

## down-stream

# pre-trained models
1. pre-trained model for jigsaw with Resnet 50 with 67% accuracy on pretext task https://mbzuaiac-my.sharepoint.com/:u:/g/personal/20020053_mbzuai_ac_ae/Ed2xPGXaqqpNuQfawHm5HvYBUbW4fL3HNLnTr9HAcrtDvQ?e=3OnR8N
2. pre-trained model of image captioning using the jigsaw extracted features https://mbzuaiac-my.sharepoint.com/:u:/g/personal/20020053_mbzuai_ac_ae/EXHOb314z-1JlFZKr-umQ8kBOl_A9Q6s3ijJxWxknnheNQ?e=2Q00iC
