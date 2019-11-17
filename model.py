from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Input, Activation, Flatten, Dropout, BatchNormalization
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Lambda, Concatenate
from tensorflow.keras.models import Model

def make_classification_model():
    inputs = Input(shape=(150, 150, 3))

    # Following the standard conv-pool scheme we construct a convolutional layer
    x = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv1')(inputs)
    # And following up with a (2,2) Max-Pool
    x = MaxPool2D()(x)
    # Batch normalization is a regularization tool (keeps output normalized e.g. [-1,1])
    x = BatchNormalization()(x)

    # We do this a few times, each time increasing the number of filters while reducing 
    # in spatial resolution:
    
    # TODO: add similar conv-pool-BN layers pattern to match the architecture below
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv4')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu', name='conv5')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    # Taking an average pool over the filter domain. The output will be 192, flattening the 
    # visual features to a long vector.
    x = GlobalAveragePooling2D()(x)

    # Finally we output a "binary" decision using two neurons with softmax activation (this is
    # done to help us in visualization, but for real classification we can use a single neuron
    # with 'sigmoid' activation)
    output = Dense(2, activation='softmax')(x)

    return Model(inputs=inputs, outputs=output)

def make_detection_model(bb_box_grid):
    classification_model = make_classification_model()
    classification_model.load_weights('cats_vs_dogs.best_weights.hdf5') 
    
    
    feature_layer = classification_model.layers[-4].output
    x = Conv2D(32, (1,1))(feature_layer) # a 1x1 conv to reduce filter depth from 192 to 32
    gathered = Flatten()(x)
    
    concat = []
    for i in range(len(bb_box_grid)):
        offset = Dense(4, activation='linear')(gathered)
        c = Dense(1, activation='sigmoid')(gathered)
        is_object = Dense(1, activation='sigmoid')(gathered)
        concat += [offset, c, is_object]
        
    x = Concatenate()(concat)
    
    return Model(inputs=classification_model.inputs, outputs=x)

def detection_loss(y_true, y_pred):
    total_loss = 0
    
    for i in range(0,6*14,6):
        total_loss += tf.keras.losses.mean_squared_error(y_true[:,i:i+4], y_pred[:,i:i+4]) + tf.keras.losses.binary_crossentropy(y_true[:,i+5], y_pred[:,i+5]) + tf.keras.losses.binary_crossentropy(y_true[:,i+4], y_pred[:,i+4])
    return total_loss