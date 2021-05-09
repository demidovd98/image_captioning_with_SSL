# Image captioning with self-supervised learning for visual feature extraction


1. The code for the project and clear instructions on running the code, 
2. A readme file (text or markdown file) to specify the function of each component (file/folder) in the codebase, 
3. List all dependencies used in the project, and provide instructions on installing any dependencies (e.g., pytorch, cudatoolkit, scikit-learn, etc.) 
4. Provide a demo file available with sample inputs and outputs.
5. Provide instructions on downloading data from publicly available links (for the datasets used in the project)
6. If a project is built on an existing code-base, it must be clearly credited and differences should be explicitly stated in the readme file. 
# Jigsaw:
Contains the files to train a jigsaw network pretext following the paper and use it for image captioning downstream task.
The impelementation of the code is improvments on Jeremalloch work in https://github.com/Jeremalloch/Semisupervised_Image_Classifier.
the differences are as follows:
1. our impelentation includes using gray images in training
2. including color jittering and make a case for gray images
3. applaying patch level normalization
4. using Alex-net as impelemented in the paper and ResNet50 for shared networks
## Dataset
The dataset used in training is MSCOCO unlabeled 2017, available on https://cocodataset.org/#download
## Code flow
#### image_preprocessing
contains image_transform.py that is used for image preprocessing for jigsaw, which include  the functions to create the croppings and to apply the color jittering
### hamming_set
Contain maximal_hamming.py which is used to generate the permutations
### to_hdf5.py
Resizes the dataset and converts it to hdf5 file to be used for training
### Datagnerator.py
Generate the puzzle patches using image_processing.py file to train the keras model 
### Transfer learning for image captioning
#### contains the files to use jigsaw on the downstream task:
1. 1_Jigsaw_generator.py extract image features for training using the model provided
2. 2_Jigsaw_model.py uses extracted features to train the captioning model
3. 3_Jigsaw_blue.py test the model using BLEU score metric
4. 4_jigsaw_test.py used to generate descriptions of provided images

Instructions on running the code is inside jigsaw directory.

