import os
import sys
import numpy as np
sys.path.append('srcs')
import utils
from keras_models import AttentionModel

if __name__ == '__main__':
    INPUT_DIR = 'data/arrays/turbofan_data'
    TASK = 'classification'
    # load data
    train_x = np.load(os.path.join(INPUT_DIR, 'train_x_w30_f14.npy'))
    val_x = np.load(os.path.join(INPUT_DIR, 'val_x_w30_f14.npy'))
    if TASK == 'classification':
        train_y = np.load(os.path.join(INPUT_DIR, 'train_y_w30_f14.npy'))
        val_y = np.load(os.path.join(INPUT_DIR, 'val_y_w30_f14.npy'))
        num_class = len(np.unique(train_y))
        # one-hot-encoding for classification task
        train_y = utils.one_hot_encode(train_y, num_class)
        val_y = utils.one_hot_encode(val_y, num_class)
    else:
        train_y = np.load(os.path.join(INPUT_DIR, 'train_y_w30.npy'))
        val_y = np.load(os.path.join(INPUT_DIR, 'val_y_w30.npy'))
        num_class = 1
    # print(train_x.shape)
    # print(train_y.shape)
    # print(val_x.shape)
    # print(val_y.shape)
    # print(np.unique(train_y, return_counts=True))
    # print(np.unique(val_y, return_counts=True))
    # model configurations
    args = {
        'window_size': train_x.shape[1],
        'feature_dim': train_x.shape[2],
        'hidden_layers': [[6, 50, 128]],
        'fully_connected_layers': [64, num_class],
        'global_pool': 'max',
        'dropout_rate': 0.1,
        'batch_normalization': True,
        'batch_size': 64,
        'epochs': 40,
        'eval_per_epoch': 5,
    }

    # instantiate model
    classification_model = AttentionModel(task=TASK, args=args)
    classification_model.train(train=[train_x, train_y],
                               val=[val_x, val_y],
                               batch_size=args['batch_size'],
                               epochs=args['epochs'],
                               eval_per_epoch=args['eval_per_epoch'])
