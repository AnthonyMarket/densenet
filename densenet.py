from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Activation, BatchNormalization, Convolution2D, Dense, \
                         Flatten, Input, AveragePooling2D, merge

growth_rate = 12 # called k in paper. They use values from 12-40
nb_channels_from_bottlenecks = 4 * growth_rate
compression_rate = 0.5  # Reduces features at transitions. Used in Densenet-C and Densenet-BC


def make_composite_layer(nb_filters, input_shape, use_bottleneck):
    '''
    Returns model object corresponding to the "Composite Function" in paper
    This is the most basic building block, and it can be thought of as a
    basic layer.

    inputs
    -------
        nb_filters: number of channels for convolutions (growth factor)
        input_shape: tuple with standard keras format and semantics for input_shapes
        use_bottleneck: Whether to include a 1x1 convolution at front of composite layer
                        NOTE: Relies on global nb_channels_from_bottlenecks when
                        bottleneck is used.

    outputs
    -------
        composite_layer: keras model object with the composite layer.
            operation is BN->optional(bottleneck+relu+BN)->relu->Conv(3,3)
    '''

    composite_layer = Sequential()
    composite_layer.add(BatchNormalization(input_shape=input_shape, mode=2))
    if use_bottleneck:
        composite_layer.add(Convolution2D(nb_channels_from_bottlenecks, 1, 1))
        composite_layer.add(Activation('relu'))
        composite_layer.add(BatchNormalization(mode=2))
    composite_layer.add(Activation('relu'))
    composite_layer.add(Convolution2D(nb_filters, 3, 3, border_mode='same'))
    return composite_layer


def make_dense_block(block_input_shape, n_layers, use_bottleneck):
    '''
    Makes block of all layers having same receptive field.
    Each later layer in this block gets inputs from all earlier layers in the block

    inputs
    -------
        block_input_shape: tuple with standard keras format and semantics for input_shapes
        n_layers: Number of composite layers to include in this dense block
                  use_bottleneck:
        use_bottleneck: Whether to include 1x1 conv bottlenecks in each composite layer

    outputs
    -------
        model: Keras model with stack of composite layers
    '''

    input = Input(shape=block_input_shape)
    layers_list = [input]
    while len(layers_list) <= n_layers:
        if len(layers_list) > 1:
            merged_earlier_layers = merge(layers_list, mode='concat')
        else:
            merged_earlier_layers = layers_list[0]
        # TODO: FIND BETTER WAY TO EXTRACT SHAPE
        merged_layers_shape = merged_earlier_layers.get_shape().as_list()[1:]
        new_layer = make_composite_layer(growth_rate,
                                         merged_layers_shape,
                                         use_bottleneck)(merged_earlier_layers)
        layers_list.append(new_layer)
    model = Model(input=input, output = layers_list[-1])
    return(model)


def make_transition_layer(input_shape):
    '''
    Creates transition layer that increases receptive field. There are no densenet connections
    across transition layers.  The transition layers are BN->Conv(1,1)->AveragePool(2,2)

    inputs
    ------
        input_shape: tuple with standard keras format and semantics for input_shapes

    outputs
    -------
        transition_layer: Keras model implementing the transition layer
    '''

    nb_filters = int(compression_rate * growth_rate)
    transition_layer = Sequential()
    transition_layer.add(BatchNormalization(input_shape=input_shape, mode=2))
    transition_layer.add(Convolution2D(nb_filters, 1, 1))
    transition_layer.add(AveragePooling2D(pool_size=(2, 2)))
    return(transition_layer)

def get_output_shape(model):
    '''
    Convenience function to get output_shape of model. Excludes observation cound

    inputs
    ------
        model: Keras model to get output shape from

    outputs
    -------
        _: tuple with format appropriate for input_shape into next layer
    '''
    return(model.get_output_shape_at(-1)[1:])

def make_model(train_data_shape,
               layer_counts_between_transitions,
               nb_classes,
               use_bottleneck=True):
    '''
    Create a densenet model. Some model parameters are hardcoded to values from original paper

    inputs
    ------
        train_data_shape: Shape of training data. Following format/semantics for keras input shapes
        layer_counts_between_transitions: List of integers, each item being the number of composite
                                          layers in that denseblock.  The list order matches the order
                                          that layers are added to the network.
        nb_classes: Number of classes in prediction target
        use_bottleneck: Boolean indicating whether to include bottlenecks in each composite layer

    outputs
    -------
        model: Compiled (but not fitted) densenet model
    '''

    model = Sequential()
    model.add(Convolution2D(16, 1, 1, border_mode='same', input_shape=train_data_shape))
    for layer_count in layer_counts_between_transitions:
        current_shape = get_output_shape(model)
        new_dense_block = make_dense_block(current_shape, layer_count, use_bottleneck)
        model.add(new_dense_block)
        current_shape = get_output_shape(model)
        new_transition_layer = make_transition_layer(input_shape=current_shape)
        model.add(new_transition_layer)
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return(model)

def fit_model(model, X_train, y_train, X_test, y_test, nb_epoch, batch_size):
    '''
    Fits model with augmented image data (standard CIFAR10 augmentation scheme)

    inputs
    ------
        model: compiled model to be fit
        X_train, y_train, X_test, y_test: numpy arrays used to fit data.
                                          test data is used for validation_data
        nb_epoch: number of training epochs
        batch_size: training batch size


    outputs
    -------
        model: fitted keras model object
    '''

    datagen = ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True)
    datagen.fit(X_train)
    model.fit_generator(datagen.flow(X_train, y_train,
                                     batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, y_test))
    return(model)
