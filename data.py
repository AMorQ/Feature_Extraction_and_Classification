import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
#print(tf.__version__)
import numpy as np
import os
#import tensorflow_datasets as tfds
from model_conv import create_model
from keras.optimizers import Adam
from keras.models import Model
from mlflow_logging import MLFlowLogger
from data_utils import resizing, normalize_img, normalize_it


class DataGenerator:
    """
    Object to obtain the patches and labels.
    """
    def __init__(self, config):
        """
        Initialize data generator object
        :param config: dict containing config
        """
        self.config = config
        self.data_config = config["data"]
        self.model_config = config["model"]

        self.ds_train, self.ds_test, self.features_fine_train, self.features_fine_test, self.labels_train, self.labels_test = self.data_generator()

    def data_generator(self):

        directory = self.data_config['dataset_dir']
        batch_size = self.model_config['batch_size']
        image_size = self.data_config['image_size']
        final_size = self.data_config['final_size']

        dataset_val = tf.keras.preprocessing.image_dataset_from_directory(directory, labels='inferred', label_mode='int', image_size=(image_size, image_size), shuffle=True, seed=123, validation_split=0.2, subset='validation', batch_size=batch_size)
        dataset_train = tf.keras.preprocessing.image_dataset_from_directory(directory, labels='inferred', label_mode='int', image_size=(image_size, image_size), shuffle=True, seed=123, validation_split=0.2, subset='training', batch_size=batch_size)

        #resize
        if image_size > final_size:
            dataset_train = resizing(dataset_train, final_size)
            dataset_val = resizing(dataset_val, final_size)

        #normalize
        ds_train = dataset_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = dataset_val.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #aquí tenía un problema si definía la función normalize_img en la clase DataGenerator

        ds_train = ds_train.cache()
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

        if self.model_config['mode'] == 'feature_extraction':

            # MODEL_CONV+TOP WITH FINE TUNING
            #todo: quiero trasladar esto al programa del modelo y meter Adam en config
            optim = Adam(learning_rate=self.model_config['learning_rate']) #SI SUBO MUCHO LA L_R TENGO UN LOSS ALTÍSIMO
            print("Load classification model")
            model_fine = create_model((512, 512, 3), 4, optimizer=optim, fine_tune=5)
            model_fine.fit(ds_train, epochs=self.model_config['epochs'])
            classification_fine = model_fine.evaluate(ds_test, return_dict=True)

            logger = MLFlowLogger(self.config)
            logger.metrics_logging('feature_extraction', classification_fine)
            # logger.data_logging(ds_train.get_train_data_statistics())

            # extract features
            model_fine_features = Model(inputs=model_fine.inputs, outputs=model_fine.layers[20].output)
            features_fine_test = model_fine_features.predict(ds_test)
            features_fine_train = model_fine_features.predict(ds_train)
            np.save(os.path.join(self.data_config['output_dir'], 'features_fine_train.npy'), features_fine_train)
            np.save(os.path.join(self.data_config['output_dir'], 'features_fine_test.npy'), features_fine_test)
        # print(features_fine_train.shape)# y este número de imágenes?

    #image_batch_train, label_batch_train = next(iter(ds_train))
    #image_batch_val, label_batch_val = next(iter(ds_test))
    #print("Image batch shape (train): ", image_batch_train.shape)
    #print("Label batch shape (train): ", label_batch_train.shape)

    #MODEL_CONV+TOP WITHOUT FINE TUNING
    #model_conv = create_model((512, 512, 3), 4)
    #conv_classification_test = model_conv.evaluate(ds_test)  # aleatoric weights, low performance that we want to improve
    #conv_classification_train = model_conv.evaluate(ds_train)  # it is done by batches
    #model_conv.fit(ds_train, epochs=5) #mejora con el número de épocas aún más
    #conv_classification_test = model_conv.evaluate(ds_test) #aleatoric weights, low performance that we want to improve
    #conv_classification_train = model_conv.evaluate(ds_train) #it is done by batches
    #conv_fit_classification_test = model_conv.evaluate(ds_test)
    #model_conv_features = Model(inputs=model_conv.inputs, outputs=model_conv.layers[20].output)
    #print(model_conv_features.summary())
    #features_conv_test = model_conv_features.predict(ds_test)
    #features_conv_train = model_conv_features.predict(ds_train)
    #print(features_conv_test.shape)
    #np.save(os.path.join(args.output_dir, 'features_conv_test.npy'), features_conv_test)
    #print('\nData summary:\n', features_conv_test)
    #np.save(os.path.join(args.output_dir, 'features_conv_train.npy'), features_conv_train)
    #print('\nData summary:\n', features_conv_train)

    #feature_array = np.load('features_fine.npy')

        elif self.model_config['mode'] == 'classification':
            features_fine_train = np.load(os.path.join(self.data_config['output_dir'], 'features_fine_train.npy'))
            features_fine_test = np.load(os.path.join(self.data_config['output_dir'], 'features_fine_test.npy'))

        features_fine_train = normalize_it(features_fine_train)
        features_fine_test = normalize_it(features_fine_test)

        labels_train = np.asarray(np.concatenate([y for x, y in ds_train], axis=0))
        #todo:cómo me aseguro de que las etiquetas estén ordenadas??
        labels_test = np.asarray(np.concatenate([y for x, y in ds_test], axis=0))
    #features_fine_train = features_conv_train
    #features_fine_test = features_conv_tes
        return ds_train, ds_test, features_fine_train, features_fine_test, labels_train, labels_test #, classification_fine






