# Image captioning with self-supervised learning for visual feature extraction


1. The code for the project and clear instructions on running the code, 
2. A readme file (text or markdown file) to specify the function of each component (file/folder) in the codebase, 
3. List all dependencies used in the project, and provide instructions on installing any dependencies (e.g., pytorch, cudatoolkit, scikit-learn, etc.) 
4. Provide a demo file available with sample inputs and outputs.
5. Provide instructions on downloading data from publicly available links (for the datasets used in the project)
6. If a project is built on an existing code-base, it must be clearly credited and differences should be explicitly stated in the readme file. 
## Jigsaw:
Contains the files to train a jigsaw network pretext and use it for image captioning downstream task.
The impelementation of the code is improvments on Jeremalloch work in https://github.com/Jeremalloch/Semisupervised_Image_Classifier.
the differences are as follows:
1. our impelentation includes using gray images in training
2. including color jittering and make a case for gray images
3. applaying patch level normalization
4. using Alex-net as impelemented in the paper and ResNet50 for networks backbone

#### image_preprocessing
contains image_transform.py that is used for image preprocessing for jigsaw, to create the croppings and apply color jittering
#to_hdf5.py

### Datagnerator

