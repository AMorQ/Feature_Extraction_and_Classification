import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import random

def resizing(dataset, final_size):
    size = [final_size, final_size]
    dataset_final = dataset.map(lambda x, y: (tf.image.resize(x, size, method='bicubic'), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #he tenido que cambiar algo la sintaxis, porque ahora estoy usando keras desde tf, por problemas de compatibilidad
    return dataset_final

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

def normalize_it(features_fine):
    data_mean, data_std = np.mean(features_fine, 0), np.std(features_fine, 0)
    data_std[data_std == 0] = 1.0
    features_fine = (features_fine - data_mean) / data_std
    return features_fine

# def resize_image(image, label, final_size):
#    size = [final_size, final_size]
#    return tf.image.resize(image, size, method='bicubic'), label

def visualize_images(config):
    #todo: SERÍA INTERESANTE QUE COGIERA UNA MUESTRA DE MUCHO TEJIDO Y EN LAS QUE COINCIDIERAN LA MAJ3 CON GT
    #Let's visualize some images from each class.


    labels = ['0', '3', '4', '5']
    dir_in = config['data']['dataset_dir']
    #todo: aquí tendré que ponerlo en función de args.criteria y añadir ese argumento
    #dir_in = '/data/Prostata/Images/Feature_Extraction/Images/{s}_aggregation'.format(s=str(args.criteria))

    #Get samples fom each category
    samples_dict = {}
    for label in labels:
        lista = os.listdir(os.path.join(dir_in, label))
        samples = random.sample(lista, 5)
        #mejor hago un diccionario de listas
        samples_dict[str(label)] = samples

    # Plot the samples
    f, ax = plt.subplots(4, 5, figsize=(15, 10))

    f.suptitle('Sample patches from pathological WSI')
    #random_samples = []  for i, samples in enumerate(random_samples):

    for key, value in samples_dict.items():
        for i, k in enumerate(value):
            random_samples_dir = os.path.join(dir_in, str(key))

            if key != '0':
                ki = int(key) - 2
            else:
                ki = int(key)

            ax[ki, int(i)].imshow(plt.imread(os.path.join(random_samples_dir, value[i])))
            ax[ki, int(i)].axis('off')
            ax[ki, int(i)].set_title('Grade{s}_#{d}'.format(s=str(key), d=str(i)))

        #for a in ax.flat:
        #    a.set(xlabel='Grade_{s}'.format(s=str(key)))
            #a.label_outer()
    plt.show()