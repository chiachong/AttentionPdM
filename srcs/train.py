import sys
import tsaug
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
    TASK = 'regression'
    WINDOW_SIZE = 30  # scanning window size
    FAIL_IN = 14  # turbofan engine fails in 14 cycle
    TRAIN_PROPORTION = 3 / 4  # proportion of the training data
    CLIP_RUL = 100
    MOVING_AVERAGES = []
    UNDER_SAMPLE = None
    # DROP_COLS = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_18', 'os_3',
    #              'sensor_16', 'sensor_19', 'sensor_22', 'sensor_23']
    # load and preprocess data
    data = dataset.TurbofanData(INPUR_DIR)
    data.preprocess(clip_RUL=CLIP_RUL, moving_averages=MOVING_AVERAGES)
    data.split_train_val(TRAIN_PROPORTION)
    if TASK == 'classification':
        arrays = data.arrays_for_classification(WINDOW_SIZE, FAIL_IN)
        train_x, train_y, test_x, test_y, val_x, val_y = arrays
        indexes_0 = np.where(train_y == 0)[0]
        indexes_1 = np.where(train_y == 1)[0]
        np.random.seed(1234)
        np.random.shuffle(indexes_0)
        # random augment
        train_x_1 = train_x[indexes_1]
        train_y_1 = train_y[indexes_1]
        augmenters = (
            tsaug.AddNoise(scale=0.02)
            + tsaug.Convolve()
            + tsaug.Pool()
            + tsaug.Drift(max_drift=0.5)
            # + tsaug.Dropout(p=0.1, fill=float("nan"), per_channel=True)
        )
        aug = augmenters.augment(train_x_1)
        train_x = np.concatenate((train_x[indexes_0[:UNDER_SAMPLE]], train_x_1, aug))
        train_y = np.concatenate((train_y[indexes_0[:UNDER_SAMPLE]], train_y_1, train_y_1))
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
    else:  # regression
        arrays = data.arrays_for_regression(WINDOW_SIZE)
        train_x, train_y, test_x, test_y, val_x, val_y = arrays
        if UNDER_SAMPLE is not None:
            indexes_clip = np.where(train_y == CLIP_RUL)[0]
            indexes_unclip = np.where(train_y < CLIP_RUL)[0]
            np.random.seed(1234)
            np.random.shuffle(indexes_clip)
            train_x = np.concatenate((train_x[indexes_clip[:UNDER_SAMPLE]], train_x[indexes_unclip]))
            train_y = np.concatenate((train_y[indexes_clip[:UNDER_SAMPLE]], train_y[indexes_unclip]))
        num_class = 1
        experiment_id = '0'

    # experiment and model settings
    args = {
        'window_size': WINDOW_SIZE,
        'feature_dim': train_x.shape[2],
        # 'hidden_layers': [[4, 32, 64], [4, 32, 64], [4, 32, 64]],  # best
        'hidden_layers': [[4, 32, 50], [4, 32, 50], [4, 32, 50]],
        'fully_connected_layers': [128, num_class],
        'global_pool': 'max',
        'dropout_rate': 0.1,
        'batch_normalization': True,
        'batch_size': 64,
        'epochs': 50,
        'eval_per_epoch': 5,
        'learning_rate_decay': 0.01,
    }
    # log experiment and model settings
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param('task', TASK)
        mlflow.log_param('train_proportion', TRAIN_PROPORTION)
        mlflow.log_param('clip_RUL', CLIP_RUL)
        mlflow.log_param('moving_averages', MOVING_AVERAGES)
        mlflow.log_param('under_sample', UNDER_SAMPLE)
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
            loss, mae, rmse = pdm_model.test(test_x, test_y)
            # log model performance
            mlflow.log_metric('test_loss', loss)
            mlflow.log_metric('test_mean_absolute_error', mae)
            mlflow.log_metric('test_root_mean_error', rmse)
        else:
            precision, recall, f1 = pdm_model.test(test_x, test_y)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('f1-score', f1)
