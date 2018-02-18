import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import applications
from keras.models import Model, load_model

img_width, img_height = 256, 256
epochs = 30
train_samples = 31520
validation_samples = 10848
test_samples = 10816
batch_size = 32
exp_url = '/home/tarek/Downloads/patches/expp2/'
train_data_dir = str(exp_url) + 'train'
validation_data_dir = str(exp_url) + 'validation'
test_data_dir = str(exp_url) + 'test'


def create_bottleneck():
    # used to rescale the pixel values from [0, 255] to [0, 1] interval
    datagen = ImageDataGenerator(rescale=1. / 255)

    # VGG16 model is available in Keras

    model_vgg = applications.VGG16(include_top=False, weights='imagenet')

    # Using the VGG16 model to process samples

    train_generator_bottleneck = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    validation_generator_bottleneck = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    # This is a long process, so we save the output of the VGG16 once and for all.

    bottleneck_features_train = model_vgg.predict_generator(train_generator_bottleneck, train_samples // batch_size)
    np.save(open(str(exp_url) + 'models/bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

    bottleneck_features_validation = model_vgg.predict_generator(validation_generator_bottleneck,
                                                                 validation_samples // batch_size)
    np.save(open(str(exp_url) + 'models/bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)


# Now we can load it...

def load_bottleneck():
    train_data = np.load(open(str(exp_url) + 'models/bottleneck_features_train.npy', 'rb'))
    train_labels = np.array([0] * (train_samples // 2) + [1] * (train_samples // 2))

    validation_data = np.load(open(str(exp_url) + 'models/bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array([0] * (validation_samples // 2) + [1] * (validation_samples // 2))

    print('train_data.shape', train_data.shape)
    print('train_labels', train_labels.shape)

    print('validation_data.shape', validation_data.shape)
    print('validation_labels', validation_labels.shape)
    # And define and train the custom fully connected neural network :

    model_top = Sequential()
    model_top.add(Flatten(input_shape=train_data.shape[1:]))
    model_top.add(Dense(256, activation='relu'))
    model_top.add(Dropout(0.5))
    model_top.add(Dense(1, activation='sigmoid'))

    model_top.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model_top.fit(train_data, train_labels,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(validation_data, validation_labels))

    # The training process of this small neural network is very fast : ~2s per epoch

    model_top.save_weights(str(exp_url) + 'models/bottleneck_30_epochs.h5')
    # model_top.load_weights(str(exp_url) + 'models/bottleneck_30_epochs.h5')

    # ### Bottleneck model evaluation
    print(model_top.evaluate(validation_data, validation_labels))


def test():
    model = load_model(str(exp_url) + 'models/theultimate.h5')
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode=None,  # only data, no labels
        shuffle=False)  # keep data in same order as labels
    print(generator.class_indices)
    # test_samples = 64
    probabilities = model.predict_generator(generator, test_samples // batch_size)
    # print('probabilities', probabilities)
    from sklearn.metrics import confusion_matrix

    # y_true = np.array([0] * 1000 + [1] * 1000)
    y_true = np.array([0] * (test_samples // 2) + [1] * (test_samples // 2))
    y_pred = probabilities > 0.5

    confusion_mtx = (confusion_matrix(y_true, y_pred))
    print(confusion_mtx)
    tn = confusion_mtx[0, 0]
    fp = confusion_mtx[0, 1]
    fn = confusion_mtx[1, 0]
    tp = confusion_mtx[1, 1]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    print("Accuracy: %.2f" % accuracy)
    print("Recall: %.2f" % recall)
    print("Precision: %.2f" % precision)
    print("F1 Score: %.2f" % f1_score)


def fine_tune():
    # Start by instantiating the VGG base and loading its weights.
    epochs = 10

    model_vgg = applications.VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    # Build a classifier model to put on top of the convolutional model. For the fine tuning, we start with a fully trained-classifer. We will use the weights from the earlier model. And then we will add this model on top of the convolutional base.

    top_model = Sequential()
    top_model.add(Flatten(input_shape=model_vgg.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    top_model.load_weights(str(exp_url) + 'models/bottleneck_30_epochs.h5')

    # model_vgg.add(top_model)
    model = Model(inputs=model_vgg.input, outputs=top_model(model_vgg.output))

    # For fine turning, we only want to train a few layers.  This line will set the first 25 layers (up to the conv block) to non-trainable.

    for layer in model.layers[:15]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # prepare data augmentation configuration  . . . do we need this?
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    # fine-tune the model
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_samples // batch_size)

    model.save_weights(str(exp_url) + 'models/finetuning_30epochs_vgg.h5')
    model.save(str(exp_url) + 'models/theultimate.h5')

    # ### Evaluating on validation set

    # Computing loss and accuracy :

    print(model.evaluate_generator(validation_generator, validation_samples))


def main():
    # create_bottleneck()
    # load_bottleneck()
    # fine_tune()
    test()


if __name__ == "__main__":
    main()
