import os
import argparse
import yaml
from data import DataGenerator
from model_conv import create_model, model_conv_classificationSVGP
from keras.optimizers import Adam
from keras.models import Model
import numpy as np
from mlflow_logging import MLFlowLogger
from predata import main_predata
import tensorflow as tf
from data_utils import visualize_images


def load_configs(args):
    with open(args.default_config) as file:
        config = yaml.full_load(file)

    return config

def some_args():

    parser = argparse.ArgumentParser(description='Parameters specification for feature extraction and classification ')

    print('Load configuration')
    parser.add_argument("--default_config", "-dc", type=str,
                        default="/data/Prostata/Images/Methods/Feature_Extraction/config.yaml",
                        help="Config path (yaml file expected) to default config.")

    return parser.parse_args()

def main(config):

    visualize_images(config)

    if config['data']['dataset_split'] == True:
        main_predata()
        #todo: modular esto

    data_feature_gen = DataGenerator(config)
    #todo: si no le haces fine tune a la red para qué haces este split??

    #np.save(os.path.join(self.data_config['output_dir'], 'features_fine_train.npy'),data_feature_gen.features_fine_train)
    #print('\nData summary:\n', data_feature_gen.features_fine_train)

    features_fine_train = data_feature_gen.features_fine_train
    features_fine_test = data_feature_gen.features_fine_test
    labels_train = data_feature_gen.labels_train
    labels_test = data_feature_gen.labels_test

    model_svgp, Y_pred, dicto = model_conv_classificationSVGP(features_fine_train, labels_train, 50, features_fine_test,
                                                       labels_test)
    #todo: Y_pred debería sacarla del modelo

    #MLFLOW

    logger = MLFlowLogger(config)
    logger.mlflow_logging()
    #logger.data_logging(ds_train.get_train_data_statistics())

    #todo: reformat dicto to correctly log to mlflow (should be done in metrics)

    values = {}
    for key, value in dicto.items():
        if key == '0' or key == '1' or key == '2' or key == '3':
            for k, v in value.items():
                ka = str(key) + '_' + k
                values[ka] = value[k]
                #value.pop(k)
                #del value[k]
            dicto[key] = values
            logger.metrics_logging('metrics_{s}'.format(s=str(key)), dicto[key])
        else:
            continue




if __name__ == '__main__':
    args = some_args()
    print('Arguments:')
    print(args)
    config = load_configs(args)
    os.makedirs(config['data']['output_dir'], exist_ok=True)
    main(config)