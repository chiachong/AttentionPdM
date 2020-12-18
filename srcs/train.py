import os
import sys
import numpy as np
sys.path.append('srcs')
import utils
import dataset
from keras_models import AttentionModel

if __name__ == '__main__':
    INPUR_DIR = 'data/CMAPSSData'
    TASK = 'regression'
    WINDOW_SIZE = 30  # scanning window size
    FAIL_IN = 14  # turbofan engine fails in 14 cycle
    TRAIN_PROPORTION = 3 / 4  # proportion of the training data
    CLIP_RUL = 125
    # load and preprocess data
    data = dataset.TurbofanData(INPUR_DIR)
    data.preprocess(clip_RUL=CLIP_RUL)
    data.split_train_val(TRAIN_PROPORTION)
    if TASK == 'classification':
        arrays = data.arrays_for_classification(WINDOW_SIZE, FAIL_IN)
        train_x, train_y, test_x, test_y, val_x, val_y = arrays
        num_class = len(np.unique(train_y))
        # one-hot-encoding for classification task
        train_y = utils.one_hot_encode(train_y, num_class)
        test_y = utils.one_hot_encode(test_y, num_class)
        val_y = utils.one_hot_encode(val_y, num_class)
    else:
        arrays = data.arrays_for_regression(WINDOW_SIZE)
        train_x, train_y, test_x, test_y, val_x, val_y = arrays
        num_class = 1

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
        'epochs': 100,
        'eval_per_epoch': 3,
    }

    # instantiate model
    pdm_model = AttentionModel(task=TASK, args=args)
    try:
        pdm_model.train(train=[train_x, train_y],
                        val=[test_x, test_y],
                        batch_size=args['batch_size'],
                        epochs=args['epochs'],
                        eval_per_epoch=args['eval_per_epoch'])
    except KeyboardInterrupt:
        pdm_model.test(val_x, val_y)
