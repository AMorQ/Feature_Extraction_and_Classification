import mlflow


class MLFlowLogger:
    """
    Object to collect configuration parameters, metrics and artifacts and log them with mlflow.
    """
    def __init__(self, config):
        self.data_config = config['data']
        self.model_config = config['model']
        mlflow.end_run()  # bc (or in case) there is an active run already
        mlruns_folder = '/work/work_alba/mlflow_server'  # 'mlruns'
        experiment_name = 'feature_extraction'
        # current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
        run_name = 'feature_extraction_annotated'

        mlflow.set_tracking_uri(mlruns_folder)
        experiment_id = mlflow.set_experiment(experiment_name=experiment_name)
        # eperiment_id = current_experiment['experiment_id']
        mlflow.start_run(run_name=run_name)  # can be used create_run but it may not
        # todo: he quitado la parte de experiment_id=experiment_id con la idea de que me lo coja automáticamente
        # he cambiado la versión de mlflow y tengo un error sintáctico
        # change the 'active_run' used by mlflow.log_param()
    def mlflow_logging(self):

    #todo: hacer bien este logging. Me crea una carpeta que tengo que mover al 0, porque es donde apunta mi mlflow. Pero la
    #carpeta de este experimento se me ha creado con un número raro!=0. La he copiado a 0 y me aparece ene mlflow con otro nombre, así que genial

    #TOTO: NO, MLFLOW APUNTA BIEN HACIA EL EXPERIMENT_ID, QUE AHORA ES ESE NUMERO LARGO

        mlflow.log_param('batch size', self.model_config['batch_size'])
        mlflow.log_param('image size', self.data_config['image_size'])
        mlflow.log_param('final size', self.data_config['final_size'])
        mlflow.log_param('learning rate', self.model_config['learning_rate'])
        #mlflow.log_params(dictionary)


        #mlflow.log_metric('classes', len(dataset_train.class_names))
        #mlflow.log_metric('number of elements for training', len(dataset_train) * 32)


        #mlflow.log_metric('loss_fine', classification_fine['loss'])
        #mlflow.log_metric('acc_fine', classification_fine['sparse_categorical_accuracy'])
        #mlflow.log_metrics(classification_SVGP, Y_pred)
        #mlflow.log_metric('classification loss_fine', classification_fine['loss'])
        #mlflow.log_metric('classification accuracy_fine', classification_fine['sparse_categorical_accuracy'])

    def data_logging(self, data_dict):
        mlflow.log_params(data_dict)
    def metrics_logging(self, name:str, report: dict):
        #me gustaría guardarlos con un nombre
        #mlflow.log_metric('loss_fine', classification['loss'])
        #mlflow.log_metric('acc_fine', classification['sparse_categorical_accuracy'])
        mlflow.log_metrics(report)
