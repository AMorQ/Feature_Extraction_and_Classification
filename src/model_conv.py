import os
import tensorflow as tf
#from tensorflow.keras.models import Model
from keras.models import Model #ya he descargado keras, creo que sin incompatibilidades
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D
import numpy as np
import gpflow
import cv2
from metrics import metrics
#from SVGP_utils import model_train_SVGP
#import pickle
from SVGP_utils import plot_model, save_model
from data_utils import normalize_it
from gpflow.utilities import print_summary
#from multiclass_classification import plot_posterior_predictions
#from scipy.cluster.vq import kmeans
#from data import normalize_it
#import IPython

def create_model(input_shape, n_classes: int, optimizer='rmsprop', fine_tune: int = 0):
    """
    Compiles a model integrated with VGG16 pretrained layers

    input_shape: tuple - the shape of input images (width, height, channels)
    n_classes: int - number of classes for the output layer
    optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
    fine_tune: int - The number of pre-trained layers to unfreeze.
                If set to 0, all pretrained layers will freeze during training
    """

    # Pretrained convolutional layers are loaded using the weights obtained with Imagenet DB.
    # Include_top is set to False, in order to exclude the model's fully-connected layers, the ones for classification
    conv_base = VGG16(include_top=False,
                      weights='imagenet',
                      input_shape=input_shape)
                      #input_tensor=Input(shape=(2048, 2048, 3))

    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
        # Create a new 'top' of the model (i.e. fully-connected layers).
        # This is 'bootstrapping' a new top_model onto the pretrained layers.

        # classification model
        top_model = conv_base.output
        top_model = MaxPooling2D(name="pooling", pool_size=(16, 16))(conv_base.output)
        top_model = Flatten(name="flatten")(top_model)
        top_model = Dense(4096, activation='relu')(top_model)
        top_model = Dense(1072, activation='relu')(top_model)
        top_model = Dropout(0.2)(top_model)
        output_layer = Dense(n_classes)(top_model)

        # Group the convolutional base and new fully-connected layers into a Model object.
        model = Model(inputs=conv_base.input, outputs=output_layer)
    else:
        for layer in conv_base.layers:
            layer.trainable = False

        # classification
        top_model = MaxPooling2D(name="pooling", pool_size=(16, 16))(conv_base.output)
        top_model = Flatten(name='flatten')(top_model)
        top_model = Dense(4096, activation='relu')(top_model)
        top_model = Dense(1072, activation='relu')(top_model)
        top_model = Dropout(0.2)(top_model)
        output_layer = Dense(n_classes)(top_model)
        model = Model(inputs=conv_base.input, outputs=output_layer)

    # Compiles the model for training.
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.summary()

    return model

def model_conv_classificationSVGP(Xtrain, Ytrain, num_ind:int, Xtest, Ytest):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, _, Z = cv2.kmeans(Xtrain.astype('float32'), num_ind, None, criteria, 10, flags)
    #Z = kmeans(X_train_sub, num_inducing)[0]
    Z = normalize_it(Z)
    print("Inducing points are computed")

    Ytrain = (Ytrain.reshape(-1, 1)).astype('float64')
    Xtrain = (Xtrain.reshape(-1, 512)).astype('float64')


    #X_train_sub, X_valid, Y_train_sub, Y_valid = train_test_split(Xtrain, Ytrain, test_size=0.2, random_state=42)
 

    invlink = gpflow.likelihoods.RobustMax(4)
    model_svgp = gpflow.models.SVGP(gpflow.kernels.SquaredExponential(),
                                    gpflow.likelihoods.MultiClass(4, invlink=invlink),
                                    np.float64(Z.reshape(-1, 512)))
    print_summary(model_svgp)
    # must be of the dimension of data (variance and lengthscales)

    #first we put the inducing points fixed
    gpflow.set_trainable(model_svgp.inducing_variable, False)

   
    #minibatch_size = 50 if Xtrain.shape[0] > 50 else None
    #model_svgp.X.set_batch_size(minibatch_size)
    #model_svgp.Y.set_batch_size(minibatch_size)

    Xtrain = normalize_it(Xtrain)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model_svgp.training_loss_closure((Xtrain.astype(np.float64), Ytrain.astype(np.float64))), model_svgp.trainable_variables)
    #until convergence

    Xtest = normalize_it(Xtest)

    print_summary(model_svgp, 'notebook')
    parameter_dict = gpflow.utilities.parameter_dict(model_svgp)

    #plot_model(model_svgp, opt)

    #PREDICTIONS
    Fsamples = model_svgp.predict_f_samples(Xtest.astype('float64'), 4).numpy().squeeze().T #.numpy() converts tensor from tf to numpy arrays
    Psamples = model_svgp.likelihood.invlink(Fsamples)
    Y_pred = np.argmax(Psamples, axis=1)
    #Ypred = model_svgp.predict_y(Xtest.astype('float64'))

    #probs = model_svgp.predict_y(Xtest.astype('float64'))[0]#todo: por qué me dan todas iguales??
    #preds = np.argmax(probs, axis=1)

    #SAVE METRICS
    dicti = metrics(Ytest, Y_pred)

    #todo: casobinario???
    #F_mean, _ = model_svgp.predict_f(Xtest.astype('float64'), 4) #me predice el valor de la función latente en los puntos Xtest
    #P = model_svgp.likelihood.invlink(F_mean)
    #print(P)

    #-----------------------------------------------------------------------------------
    #m = gpflow.models.SVGP(gpflow.kernels.RBF(variance=2.0, lengthscales=2.0),gpflow.likelihoods.Softmax(4), np.float64(Z.reshape(-1, 512)))  # , num_latent=4)#, minibatch_size=minibatch_size)

    m = gpflow.models.SVGP(gpflow.kernels.RBF(),
                           gpflow.likelihoods.Softmax(4), np.float64(Z.reshape(-1, 512)))#, num_latent=4)#, minibatch_size=minibatch_size)
    print_summary(m)
    #other_model = model_train_SVGP(m, Xtrain, Xtest, Ytest, num_inducing=50)

    #probs = other_model.predict_y(Xtest)[0]
    #preds = np.argmax(probs, axis=1)

    #---------------------------------------------------------------------------------------
    dir2Save = './models/'
    if not os.path.exists(dir2Save):
        os.makedirs(dir2Save)
    path = dir2Save + 'model_svgp_MV_robustmax.pkl'
    #with open(path, 'wb') as fp:
    #    pickle.dump(model_svgp.read_trainables(), fp)

    save_model(model_svgp, Xtest, path)

    return model_svgp, Y_pred, dicti

#def SVGPCR_model():

