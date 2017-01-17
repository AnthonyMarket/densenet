from densenet import make_model, fit_model
from keras.datasets import cifar10
from keras.utils import np_utils


def get_data():
    '''
    Downloads and normalizes cifar10 data. Uses standard train/test split
    '''
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # Convert class vectors to binary class matrices.
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    return((X_train, Y_train), (X_test, Y_test))

if __name__ == "__main__":
    nb_epoch = 100
    batch_size = 64
    data_augmentation = True
    print("Loading Data")
    ((X_train, y_train), (X_test, y_test)) = get_data()
    train_data_shape = X_train.shape[1:]
    model = make_model(train_data_shape, layer_counts_between_transitions = [13, 13, 13], nb_classes=10)
    print(model.summary())
    model = fit_model(model, X_train, y_train, X_test, y_test, nb_epoch, batch_size)


    print("Saving model to disk")
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
