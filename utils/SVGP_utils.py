import time
import gpflow
import numpy as np
from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

@tf.function
def optimization_step(opt, model, loss, data_iterator):

    #opt.minimize(model.training_loss, var_list=model.trainable_variables)
    opt.minimize(model.training_loss(data_iterator), var_list=model.trainable_variables)


def model_train_SVGP(m, Xtrain, Xtest, Y_valid, num_inducing):
    # Iters
    iters = [1000, 20000, 10000]
    trainTime_e = []
    ELBO = []
    iter_notif = 100
    # Sesion
    #sess = tf.Session() #is deprecated, iniciatialization automatically
    optop = Adam(learning_rate=0.001)
    optop2 = Adam(learning_rate=0.0001)

    #loss = -m.likelihood_tensor #versión anterior
    #data_iterator = iter(tuple([a for a in Xtrain]))
    data_iterator = iter(Xtrain)
    #tengo un problema aquí porque tengo una tuple de logitud 2 y no me coge bien los datos
    loss = m.training_loss_closure(data_iterator)
    #data_iterator = iter(dataset) puedo poner en dataset Xtrain?
    #optop = tf.train.Adam(0.001).minimize(loss) #no encuentra Adam

    #todo: hay otra forma de hacer esto más interesante
    #todo: cómo digo que me entrene tb las Z
    #try:
    #    for layer in m.layers[:-1]:
    #        layer.feature.Z.set_trainable(False)
    #except:
    #    m.feature.Z.set_trainable(False)
#todo: poner bien los datos para loss_trainig_closure()
    """
print("0 training starts:")
    start = time.time()
    for _ in range(iters[0]):

        optimization_step(optop, m, loss, data_iterator)
        #sess.run((m.likelihood_tensor, optop))
    print(m.as_pandas_table())
    trainTime_e.append(time.time() - start)
    print("Finished:", trainTime_e[-1])
    #m.anchor(sess)
    ELBO.append(np.mean(np.array([m.compute_log_likelihood() for a in range(10)])))
    print('======================training 0 : =======================')
    print('ELBO: ', ELBO[-1], '\n')

    #metrics = get_metrics(X_valid, y_valid, m)
    #print(metrics[0])

    #if Xtrain.shape[0] > num_inducing:
    #    try:
    #        for layer in m.layers[:-1]:
    #            layer.feature.Z.set_trainable(True)
    #    except:
    #        m.feature.Z.set_trainable(True)

    print("1 training starts:")
    start = time.time()
    for j in range(int(iters[1] / iter_notif)):
        for _ in range(iter_notif):
            #sess.run((m.likelihood_tensor, optop))
            optimization_step(optop, m, loss)
        #m.anchor(sess)
        ELBO.append(np.mean(np.array([m.compute_log_likelihood() for a in range(10)])))
        print('======================iter, ', j, ' of ', int(iters[1] / iter_notif), ': =======================')
        print('ELBO: ', ELBO[-1], '\n')
        #metrics = get_metrics(X_valid, y_valid, m)
        #print(metrics[0])

    trainTime_e.append(time.time() - start)
    print("Finished train 2:", trainTime_e[-1])

    print("2 training starts:")
    start = time.time()
    for j in range(int(iters[2] / iter_notif)):
        for _ in range(iter_notif):
            #sess.run((m.likelihood_tensor, optop2))
            optimization_step(optop2, m, loss)
        #m.anchor(sess)
        ELBO.append(np.mean(np.array([m.compute_log_likelihood() for a in range(10)])))
        print('======================iter, ', j, ' of ', int(iters[2] / iter_notif), ': =======================')
        print('ELBO: ', ELBO[-1], '\n')
        #metrics = get_metrics(X_valid, y_valid, m)
        #print(metrics[0])

    trainTime_e.append(time.time() - start)
    print("Finished train 2:", trainTime_e[-1])
    #m.anchor(sess)
    #aquí me gustaría también sacar las predicciones si no no puedo comparar los modelos




    """

    #images_subset, labels_subset = next(iter(dataset.batch(32)))
    #images_subset = tf.reshape(images_subset, [-1, image_size])
    #labels_subset = tf.reshape(labels_subset, [-1, 1])
    #m, v = m.predict_y(images_subset)
    #preds = np.argmax(m, 1).reshape(labels_subset.numpy().shape)
    #correct = preds == labels_subset.numpy().astype(int)
    #acc = np.average(correct.astype(float)) * 100.0
    #print("Accuracy is {:.4f}%".format(acc))
    #plot_model(m, optop)
    return m

def plot_model(model:gpflow.models.GPModel, opt, X, Y) -> None:
    #X, Y = model.data #SVGP object has no attribute data
    opt.minimize(model.training_loss, model.trainable_variables)

    Xplot = np.linspace(0.0, 1.0, 200)[:, None]

    y_mean, y_var = model.predict_y(Xplot, full_cov=False)[0]
    y_lower = y_mean - 1.96 * np.sqrt(y_var)
    y_upper = y_mean + 1.96 * np.sqrt(y_var)

    _, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(X, Y, "kx", mew=2)
    (mean_line,) = ax.plot(Xplot, y_mean, "-")
    color = mean_line.get_color()
    ax.plot(Xplot, y_lower, lw=0.1, color=color)
    ax.plot(Xplot, y_upper, lw=0.1, color=color)
    ax.fill_between(
        Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color=color, alpha=0.1
    )

    Fsamples = model.predict_f_samples(Xplot, 10).numpy().squeeze().T
    plt.plot(Xplot, Fsamples, "C0", lw=0.5)

    Psamples = model.likelihood.invlink(Fsamples)

    plt.plot(Xplot, Psamples, "C1", lw=0.5)
    plt.scatter(X, Y)

    Fmean, _ = model.predict_f(Xplot)
    P = model.likelihood.invlink(Fmean)

    plt.plot(Xplot, P, "C1")
    plt.scatter(X, Y)

def save_model(model, Xtest, dir):#Xtest no está normalizado???
    model.compiled_predict_f = tf.function(
        lambda Xtest: model.predict_f(Xtest, full_cov=False),
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
    )
    model.compiled_predict_y = tf.function(
        lambda Xtest: model.predict_y(Xtest, full_cov=False),
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
    )

    tf.saved_model.save(model, dir)

    #for later loading
    """
    loaded_model = tf.saved_model.load(save_dir)

    plot_prediction(model.predict_y)
    plot_prediction(loaded_model.compiled_predict_y)
    """

"""
def plot_kernel_samples(): 
def plot_kernel_prediction(): 
def plot_kernel(): 
"""

