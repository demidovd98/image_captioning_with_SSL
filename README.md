# Image captioning with self-supervised learning for visual feature extraction


1. The code for the project and clear instructions on running the code, 
2. A readme file (text or markdown file) to specify the function of each component (file/folder) in the codebase, 
3. List all dependencies used in the project, and provide instructions on installing any dependencies (e.g., pytorch, cudatoolkit, scikit-learn, etc.) 
4. Provide a demo file available with sample inputs and outputs.
5. Provide instructions on downloading data from publicly available links (for the datasets used in the project)
6. If a project is built on an existing code-base, it must be clearly credited and differences should be explicitly stated in the readme file. 

# Sample Output of Model:

![Captions_For_Updated_Model](https://user-images.githubusercontent.com/45034431/117573575-aeb68d80-b0e9-11eb-9991-4414f0ba6307.JPG)

note that the lighter color refers to more acceptable captions, whereas the darker color shows less acceptable captions.

# Problem Statement
In this work a problem of generating a textual description for a given image, using self-supervised learned (SSL) approaches as a pre-text task is considered.
Originally, the existed solutions utilised fully supervised trained models for the part of image feature extraction. However, our experiments showed that such a complex task as image captioning requires higher level of generalisation than usual models can provide. This problem could be addressed with using self-supervised learning methods, that recently showed their ability to generalise better.
In order to explore this property of SSL approaches, we proposed and explored two solutions for the image captioning using two different self-supervised learned models, based on Jigsaw Puzzle solving and SimCLR, as a pre-text task.

# Experiments
For the sake of supervised and self-supervised pre-text tasks comparison, we provide the results of their comprehensive testing on the same downstream task, calculating a BLEU score and validation loss. Our proposed solution with SimCLR model used for image feature extraction achieved the following results: BLEU-1: 0.575, BLEU-2: 0.360, BLEU-3: 0.266, BLEU-4: 0.145, and validation loss of 3.415. These outcomes can be considered as competitive ones with the fully supervised solutions.
At the end, we also provide result of conducted ablation study for the mentioned approaches, including usage of different models and theirs optimisations\footnote{Code, pre-trained models, and instructions for them are available at
# Getting started
Create a conda environment using the dependencies file:

```sh
conda env create -f dependencies.yml
```
# Jigsaw:
Contains the files to train a jigsaw network pretext following the paper [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246) and use it for image captioning downstream task.
The impelementation of the code builds on Jeremalloch work in https://github.com/Jeremalloch/Semisupervised_Image_Classifier.
the differences are as follows:
1. using Alex-net as impelemented in the paper and ResNet50 for shared networks
2. our impelentation includes using gray images in training
3. includes color jittering
4. applaying normalization on patch level

## Dataset
The dataset used in training is MSCOCO unlabeled 2017,from https://cocodataset.org can be downoaded here  [MSCOCO unlabeled 2017](http://images.cocodataset.org/zips/unlabeled2017.zip) 
## method components:
#### image_preprocessing:
contains image_transform.py that is used for image preprocessing for jigsaw, which include  the functions to create the croppings and to apply the color jittering
### hamming_set:
Contain maximal_hamming.py which is used to generate the permutations
### to_hdf5.py:
Resizes the dataset and converts it to hdf5 file to be used for training
### Datagnerator.py:
Generate the puzzle patches using image_processing.py to train the keras model 
### Transfer learning for image captioning:
#### contains the files to use jigsaw on the downstream task:
1. Jigsaw_generator.py extract image features for training using the model provided
2. Jigsaw_model.py uses extracted features to train the captioning model
3. Jigsaw_blue.py test the model using BLEU score metric
4. jigsaw_test.py used to generate descriptions of provided images

