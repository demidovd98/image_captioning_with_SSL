# Image Captioning with Self-Supervised Learning Visual Feature Extraction
A problem of generating a textual description for a given image with using self-supervised learned approaches (SimCLR and Jigsaw Puzzle solving) as a pre-text task is considered in this work.


# Sample Outputs of the Best Trained Image Captioning Model:
![Captions_For_Updated_Model](https://user-images.githubusercontent.com/45034431/117573575-aeb68d80-b0e9-11eb-9991-4414f0ba6307.JPG)

Captions generated by the captioning generator trained with using a SimCLR self-supervised model as a pre-text task. We manually marked the generated captions quality as follows: green colour represents good quality, yellow colour - adequate quality, red - unacceptable quality.

The ablation table with specifying detailed model's parameters and theirs captions for random images is published at [Ablation table](https://mbzuaiac-my.sharepoint.com/:x:/g/personal/20020067_mbzuai_ac_ae/Eb3-nJqxjHlMiZakj-IgmsgBKOQXurV7nNxIu5FMifw9bA?e=TSw6Aw).
P. S. The random images that were used in the ablation table for evaluation of captions are available at [Images](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/20020067_mbzuai_ac_ae/EgA68ByBqzZBoGpod0LhfCwBNFeZinkpj2ZnM3UYZhzFkQ?e=PItlED)


# Problem Statement
Originally, the existed solutions utilised fully supervised trained models for the part of image feature extraction. However, our experiments showed that such a complex task as image captioning requires higher level of generalisation than usual models can provide. This problem could be addressed with using self-supervised learning methods, that recently showed their ability to generalise better.
In order to explore this property of SSL approaches, we proposed and explored two solutions for the image captioning using two different self-supervised learned models, based on Jigsaw Puzzle solving and SimCLR, as a pre-text task.


# Results
For the sake of supervised and self-supervised pre-text tasks comparison, we provide the results of their comprehensive testing on the same downstream task, calculating a BLEU score and validation loss. Our proposed solution with SimCLR model used for image feature extraction achieved the following results: BLEU-1: 0.575, BLEU-2: 0.360, BLEU-3: 0.266, BLEU-4: 0.145, and validation loss of 3.415. These outcomes can be considered as competitive ones with the fully supervised solutions.
Along with the code, we also provide pre-trained models for image captionig task, which can be used for any random image.


# Acknowledgements:
1. The impelementation of the caption generator code is adopted from the Jason's Brownlee work at [Machine learning mastery blog](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/).
2. The idea of the Jigsaw Puzzle solver implementation is adopted from the [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246). The impelementation of the code builds on Jeremalloch work in https://github.com/Jeremalloch/Semisupervised_Image_Classifier.
3. The idea of the SimCLR implementation is adopted from the [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf).

# Getting started
Create an conda environment using the dependencies ".yml" file (for a chosen SSL pre-text task):
```sh
conda env create -f dependencies.yml
```

## Datasets
1. The dataset used for training the Jigsaw Puzzle solving pre-text task is MSCOCO unlabeled 2017,from https://cocodataset.org, can be downoaded here  [MSCOCO unlabeled 2017](http://images.cocodataset.org/zips/unlabeled2017.zip) 

2. The dataset used for training the Caption generator model downstream task is Flickr8k, which can be downoaded from the shared folder [Flickr8k Dataset](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/20020067_mbzuai_ac_ae/Eph0MaHxg5FCjmMvtQf_p7sBjBf6G4_JWkCPXoawXZT9Mw?e=5EVLxf) 


# Image Captioning with the Contrastive Learning Framework

## To train a model from scratch:
The root folder contains the code and instructions of using SimCLR model as a pretext task for extracting features for the image captioning downstream task.
0. Setup environment with the provided "dependencies.yml" file.

(For each foolowing step you will ned to also provide a correct path to the dataset and any updated by training file):
1. Run "1_data_preprocessor.py" file to extract visual features from the images and textual descriptions from the descriptions in the chosen dataset. They will be put in the "features.pkl" file and "descriptions.txt" file respectively.
2. Run "2_train_IC_model.py" file to train the caption generator model with extracted in the previous step features. The trained model will be saved in the same root directory.
3. Run "3_BLEU.py" file to evaluate the BLEU score of the pre-trained model.
4. Run "4_tokenizer.py" to create a "tokenizer.pkl" file for further encoding generated words for the model while generating a sequence,
5. Run "5_test.py" file to generate a caption for any image.

## To use a pre-trained model and files:
0. Download the pre-trained model, extracted features and descriptions, and tokenizer from the shared folder [Image_Captioning with_SimCLR](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/20020067_mbzuai_ac_ae/EpbmvMjAMQlNij__vSXoOMQBdv34t5Ws47uIeUdH4LgT3A?e=xQGWWv) and put them into a "Pre-trained/" folder (to the same folder with code).
1. Run "3_BLEU.py" file to evaluate the BLEU score of the pre-trained model. (Also provide a correct path to a chosen dataset).
2. Run "5_test.py" file to generate a caption for any image.(An image should be in the same code folder, or the full path to it should be provided).


# Image Captioning with Jigsaw Puzzle Solving:
Contains the files neccessary to train jigsaw puzzle solver network following the paper [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246) and use it for image captioning downstream task.
The impelementation of the code builds on Jeremalloch work in https://github.com/Jeremalloch/Semisupervised_Image_Classifier.
the differences are as follows:
1. Using Alex-net as impelemented in the paper and ResNet50 for shared networks instead of the used Resnet34 and trivial net
2. Our impelentation includes using gray images in training
3. Includes color jittering
4. Applaying normalization on patch level

## Instructions on running the code
### Pretext Task
1. Create the HDF5 dataset (the model expects the data to be in HDF5 file) using the to_hdf5.py specifing the path and desired image size.
2. Create the desired number permutations using maximal_hamming.py in hamming_set by providing number of permutations, which will create a text file with the permutations.
3. In the main.py add the dataset path and the hamming_set (number of permutations) text file path and specify the number of permutations (max_hamming_set) to use from the permutations list (choose the same number if you want to use all the created permutations).
4. In case of using different image size, need to be specified in the dataset creation, datagenerator creation(image_size parameter) and in image_transform ( where crop size is the size of the random crop, cell size  is the size of large patch, and tile size is the final size of the patch).

#### DataGenerator.py:
Generate the puzzle patches using image_processing.py to train the keras model
#### image_preprocessing:
contains image_transform.py that is used for image preprocessing for jigsaw, which include  the functions to create the croppings and to apply the color jittering

### Downstream Task
1. Run Jigsaw_feature_extraction.py, load the trained model and spacify dataset directory, and choose one of two functions  to extract the features: 
  - Full network uses the last dense layer before the soft max of the whole architecture for features extraction, must specify the same imgae size parameters used in training.
  -  Single network intialize a ResNet50 with the trained weights and use the GAP layer for feature extraction (not recommended).
2. Run jigsaw_captions.py to extract textual descriptions in the "description.txt" file from the chosen dataset.
3. Run Jigsaw__IC_model.py to train the captioning model on the extracted features by spicifying the extracted features file location and the descriptions file.
4. Run "Jigsaw_tokenizer.py" to create a "tokenizer.pkl" file for further encoding generated words for the model while generating a sequenc
5. Run Jigsaw_test_blue.py to check the model bleu score, requires only the extracted features file location.
6. Run Jigsaw_test_images.py  to test the trained captioning model on real images, need specify image location and the used image captioning model.

### Pre-trained Models
1. Pre-trained model for jigsaw with Resnet 50 with 67% accuracy on pretext task [Jigsaw_ResNet50](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/20020053_mbzuai_ac_ae/Ed2xPGXaqqpNuQfawHm5HvYBUbW4fL3HNLnTr9HAcrtDvQ?e=3OnR8N)
2. Pre-trained model of image captioning using the jigsaw extracted features [Image_captioning_jigsaw](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/20020053_mbzuai_ac_ae/EXHOb314z-1JlFZKr-umQ8kBOl_A9Q6s3ijJxWxknnheNQ?e=2Q00iC)
