#-#-# Libs:

## Storage:
from pickle import load


## Pre-processing:
from numpy import array


## Keras:
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# Caption generator model:
from keras.models import Model # Functions with model
from keras.layers import Input # Input layer
from keras.layers import Dense # Dense layer
from keras.layers import LSTM # LSTM layer
from keras.layers import Embedding # Word Embedding layer
from keras.layers import Dropout # Dropout layer
from keras.layers.merge import add # Addition layer
from keras.callbacks import ModelCheckpoint # Checkpoints
from keras.optimizers import Adam # Optimiser

#-#-#



#-#-# Functions:

#-# Load text:
def load_text(filename):
    file = open(filename, 'r') # Open the file as read only
    text = file.read() # Read all text
    file.close() # Close the file

    return text
#-#



#-# Load a pre-defined list of image identifiers:
def load_set(filename):
    doc = load_text(filename)
    dataset = list()

    # Process line by line:
    for line in doc.split('\n'):
        # Skip empty lines
        if len(line) < 1:
            continue

        # Get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)

    return set(dataset)
#-# 



#-# Load clean descriptions into memory:
def load_clean_descriptions(filename, dataset):
    # Load document
    doc = load_text(filename)
    descriptions = dict()

    for line in doc.split('\n'):
        # Split line by white space
        tokens = line.split()

        # Split id from description
        image_id, image_desc = tokens[0], tokens[1:]

        # Skip images not in the set
        if image_id in dataset:
            # Create list
            if image_id not in descriptions:
                descriptions[image_id] = list()

            # Wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'

            # Store descriptions
            descriptions[image_id].append(desc)

    return descriptions
#-# 



#-# Load image features:
def load_image_features(filename, dataset):
    # Load all features
    all_features = load(open(filename, 'rb'))

    # Filter features
    features = {k: all_features[k] for k in dataset}
    
    return features
#-# 



#-# Covert a dictionary of clean descriptions to a list of descriptions:
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    
    return all_desc
#-# 



#-# Fit a tokenizer given caption descriptions:
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    
    return tokenizer
#-# 



#-# Calculate the length of the description with the most words:
def max_length(descriptions):
    lines = to_lines(descriptions)
    
    return max(len(d.split()) for d in lines)
#-# 



#-# Create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, images, vocab_size):
    X1, X2, y = list(), list(), list()

    # Get through each image identifier:
    for key, desc_list in descriptions.items():
        # Get through each description for the image
        for desc in desc_list:
            # Encode the sequence
            seq = tokenizer.texts_to_sequences([desc])[0]

            # Split one sequence into multiple X,y pairs:
            for i in range(1, len(seq)):
                # Split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]

                # Pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]

                # Encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                # Store sequences
                X1.append(images[key][0])
                X2.append(in_seq)
                y.append(out_seq)

    return array(X1), array(X2), array(y)
#-# 



#-# Define the captioning model
def define_model(vocab_size, max_length):
    ## Feature extractor model:
    inputs1 = Input(shape=(2048,)) # Same as output of ResNet-50 pre-last layer
    fe1 = Dropout(0.5)(inputs1) # 50 % dropout
    fe2 = Dense(256, activation='relu')(fe1) # ReLU activation

    ## Sequence model:
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    ## Decoder model:
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    ## Merge them together [image, seq] [word]:
    model = Model(inputs=[inputs1, inputs2], outputs=outputs) # Define model
    opt = Adam(learning_rate=0.01, epsilon=0.1) # Adam optimiser
    model.compile(loss='categorical_crossentropy', optimizer=opt) # Start training

    ## Print final model
    print(model.summary())

    return model
#-# 

#-#-#




#-#-# Main:

if __name__ == '__main__': # !Don't forget to change the dataset paths!
    ### Training:
    ## Load training images dataset (6K)
    filename = 'Flickr8k_text/Flickr_8k.trainImages.txt' # training dataset path
    #filename = '/home/dd/Documents/Datasets/Flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt' 
    train = load_set(filename) # Load dataset
    print('Dataset: %d' % len(train)) 


    ## Load training descriptions:
    train_descriptions = load_clean_descriptions('Pre-trained/descriptions.txt', train)
    print('Descriptions: train=%d' % len(train_descriptions))

    
    ## Load image features:
    train_features = load_image_features('Pre-trained/features.pkl', train)
    print('Images: train=%d' % len(train_features))


    ## Load Tokenizer:
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)

    # Determine the maximum sequence length
    max_length = max_length(train_descriptions)
    print('Description Length: %d' % max_length)


    ## Prepare sequences:
    X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features, vocab_size)



    ### Evaluation:
    # Load development dataset
    filename = 'Flickr8k_text/Flickr_8k.devImages.txt' # Dev dataset path
    #filename = '/home/dd/Documents/Datasets/Flickr8k/Flickr8k_text/Flickr_8k.devImages.txt'
    test = load_set(filename) # Load dataset
    print('Dataset: %d' % len(test))


    ## Load dev descriptions:
    test_descriptions = load_clean_descriptions('Pre-trained/descriptions.txt', test)
    print('Descriptions: test=%d' % len(test_descriptions))
    
    
    ## Load image features:
    test_features = load_image_features('Pre-trained/features.pkl', test)
    print('Images: test=%d' % len(test_features))


    ## Prepare sequences:
    X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features, vocab_size)



    ### Fit the model:
    # Initialise the model
    model = define_model(vocab_size, max_length)

    # Initialise checkpoint callback
    filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Fit model
    model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))

#-#-#