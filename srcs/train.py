import sys
import numpy as np
import mlflow.keras
from mlflow.tracking import MlflowClient
sys.path.append('srcs')
import utils
import dataset
from keras_models import AttentionModel

mlflow.keras.autolog()

if __name__ == '__main__':
    # settings for data preprocessing
    INPUR_DIR = 'data/CMAPSSData'
    TASK = 'classification'
    WINDOW_SIZE = 30  # scanning window size
    FAIL_IN = 14  # turbofan engine fails in 14 cycle
    TRAIN_PROPORTION = 3 / 4  # proportion of the training data
    CLIP_RUL = 110
    MOVING_AVERAGES = [5]
    # load and preprocess data
    data = dataset.TurbofanData(INPUR_DIR)
    data.preprocess(clip_RUL=CLIP_RUL, moving_averages=MOVING_AVERAGES)
    data.split_train_val(TRAIN_PROPORTION)
    if TASK == 'classification':
        arrays = data.arrays_for_classification(WINDOW_SIZE, FAIL_IN)
        train_x, train_y, test_x, test_y, val_x, val_y = arrays
        indexes = np.where(train_y == 0)[0]
        indexes1 = np.where(train_y == 1)[0]
        np.random.seed(1234)
        np.random.shuffle(indexes)
        train_x = np.concatenate((train_x[indexes[:3000]], train_x[indexes1]))
        train_y = np.concatenate((train_y[indexes[:3000]], train_y[indexes1]))
        num_class = len(np.unique(train_y))
        # one-hot-encoding for classification task
        train_y = utils.one_hot_encode(train_y, num_class)
        test_y = utils.one_hot_encode(test_y, num_class)
        val_y = utils.one_hot_encode(val_y, num_class)
        experiment_id = '1'
        # check whether there is mlflow experiment created for classification
        client = MlflowClient()
        if not any([e.experiment_id == '1' for e in client.list_experiments()]):
            mlflow.create_experiment("Classification")
    else:
        arrays = data.arrays_for_regression(WINDOW_SIZE)
        train_x, train_y, test_x, test_y, val_x, val_y = arrays
        num_class = 1
        experiment_id = '0'

    # experiment and model settings
    args = {
        'window_size': WINDOW_SIZE,
        'feature_dim': train_x.shape[2],
        # 'hidden_layers': [[4, 32, 64], [4, 32, 64], [4, 32, 64]],  # best
        'hidden_layers': [[4, 32, 50], [4, 32, 50], [4, 32, 50]],
        'fully_connected_layers': [64, num_class],
        'global_pool': 'max',
        'dropout_rate': 0.1,
        'batch_normalization': True,
        'batch_size': 256,
        'epochs': 40,
        'eval_per_epoch': 5,
        'learning_rate_decay': 0.01,
    }
    # log experiment and model settings
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param('task', TASK)
        mlflow.log_param('train_proportion', TRAIN_PROPORTION)
        mlflow.log_param('clip_RUL', CLIP_RUL)
        mlflow.log_param('moving_averages', MOVING_AVERAGES)
        mlflow.log_params(args)

        # instantiate model
        pdm_model = AttentionModel(task=TASK, args=args)
        mlflow.log_param('model_type', pdm_model._model_name)
        pdm_model.train(train=[train_x, train_y],
                        val=[val_x, val_y],
                        batch_size=args['batch_size'],
                        epochs=args['epochs'],
                        eval_per_epoch=args['eval_per_epoch'],
                        learning_rate_decay=args['learning_rate_decay'])
        if TASK == 'regression':
            loss, mae = pdm_model.test(test_x, test_y)
            # log model performance
            mlflow.log_metric('test_loss', loss)
            mlflow.log_metric('test_mean_absolute_error', mae)
        else:
            precision, recall, f1 = pdm_model.test(test_x, test_y)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('f1-score', f1)
