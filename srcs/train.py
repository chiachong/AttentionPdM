import sys
import numpy as np
import mlflow.keras
from mlflow.tracking import MlflowClient
sys.path.append('srcs')
import utils
import dataset
from keras_models import AttentionModel, CuDNNGRUModel, GRUModel

mlflow.keras.autolog()

if __name__ == '__main__':
    # settings for data preprocessing
    INPUR_DIR = 'data/CMAPSSData'
    TASK = 'regression'
    WINDOW_SIZE = 30  # scanning window size
    TRAIN_PROPORTION = 3 / 4  # proportion of the training data
    CLIP_RUL = 100
    # load and preprocess data
    data = dataset.TurbofanData(INPUR_DIR)
    data.preprocess(clip_RUL=CLIP_RUL)
    data.split_train_val(TRAIN_PROPORTION)
    arrays = data.arrays_for_regression(WINDOW_SIZE)
    train_x, train_y, test_x, test_y, val_x, val_y = arrays
    num_class = 1

    # experiment and model settings
    args = {
        'window_size': WINDOW_SIZE,
        'feature_dim': train_x.shape[2],
        'hidden_layers': [[4, 32, 50], [4, 32, 50]],  # for attention model
        # 'hidden_layers': [100],  # for CuDNNGRU or GRU model
        'fully_connected_layers': [128, num_class],
        'global_pool': 'max',
        'dropout_rate': 0.1,
        'batch_normalization': True,
        'batch_size': 512,
        'epochs': 50,
        'eval_per_epoch': 5,
        'learning_rate_decay': 0.01,
    }
    # log experiment and model settings
    with mlflow.start_run():
        mlflow.log_param('task', TASK)
        mlflow.log_param('train_proportion', TRAIN_PROPORTION)
        mlflow.log_param('clip_RUL', CLIP_RUL)
        mlflow.log_params(args)

        # instantiate model
        pdm_model = AttentionModel(task=TASK, args=args)
        mlflow.log_param('model_type', pdm_model._model_name)
        # model training
        pdm_model.train(train=[train_x, train_y],
                        val=[val_x, val_y],
                        batch_size=args['batch_size'],
                        epochs=args['epochs'],
                        eval_per_epoch=args['eval_per_epoch'],
                        learning_rate_decay=args['learning_rate_decay'])
        # model evaluation using test data
        loss, mae, rmse = pdm_model.test(test_x, test_y)
        # log model performance
        mlflow.log_metric('test_loss', loss)
        mlflow.log_metric('test_mean_absolute_error', mae)
        mlflow.log_metric('test_root_mean_error', rmse)
