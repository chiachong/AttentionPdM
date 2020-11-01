import os
import sys
import numpy as np
sys.path.append('srcs')
import dataset

if __name__ == '__main__':
    INPUR_DIR = 'data/CMAPSSData'
    OUT_DIR = 'data/arrays/turbofan_data'
    os.makedirs(OUT_DIR, exist_ok=True)
    training_data_proportion = 3 / 4
    window_size = 30  # scanning window size
    fail_in = 14  # turbofan engine fails in 14 cycle

    # load data
    data = dataset.TurbofanData(INPUR_DIR)
    # preproces step: drop columns -> calculate RUL -> normalization
    #   dropped columns: 'sensor_16', 'sensor_19', 'sensor_22', 'sensor_23'
    #                    'sensor_24', 'sensor_25', 'sensor_26'
    #   normalization done by using min-max scaler
    data.preprocess()

    # split training and validation sets
    data.split_train_val(training_data_proportion)

    # generate arrays for classification task
    arrays = data.arrays_for_classification(window_size, fail_in)
    train_x, train_y, test_x, test_y, val_x, val_y = arrays
    # save arrays
    np.save(os.path.join(OUT_DIR, 'train_x_w30_f14.npy'), train_x)
    np.save(os.path.join(OUT_DIR, 'train_y_w30_f14.npy'), train_y)
    np.save(os.path.join(OUT_DIR, 'test_x_w30_f14.npy'), test_x)
    np.save(os.path.join(OUT_DIR, 'test_y_w30_f14.npy'), test_y)
    np.save(os.path.join(OUT_DIR, 'val_x_w30_f14.npy'), val_x)
    np.save(os.path.join(OUT_DIR, 'val_y_w30_f14.npy'), val_y)
    print('Finish saving data for classification.')

    # generate arrays for regression task
    arrays = data.arrays_for_regression(window_size)
    train_x, train_y, test_x, test_y, val_x, val_y = arrays
    # save arrays
    np.save(os.path.join(OUT_DIR, 'train_x_w30.npy'), train_x)
    np.save(os.path.join(OUT_DIR, 'train_y_w30.npy'), train_y)
    np.save(os.path.join(OUT_DIR, 'test_x_w30.npy'), test_x)
    np.save(os.path.join(OUT_DIR, 'test_y_w30.npy'), test_y)
    np.save(os.path.join(OUT_DIR, 'val_x_w30.npy'), val_x)
    np.save(os.path.join(OUT_DIR, 'val_y_w30.npy'), val_y)
    print('Finish saving data for regression.')
