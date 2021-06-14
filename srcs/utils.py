import os
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List


def calculate_moving_average(df, window: int, col_names: List[str]) -> dict:
    """ Calculate moving averages with given window size. """
    new_cols = {}
    for col_name in col_names:
        new_col = df[col_name].rolling(window).mean()
        new_cols[f'{col_name}_ma_{window}'] = new_col

    return new_cols


def calculate_RUL(df: pd.DataFrame, df_RUL: pd.DataFrame = None):
    """ Calculate the RUL of each row in the input df """
    # get the maximum life for each unit
    lifes = {}
    # Max life could be calculated from train df since they are
    # run-to-failure data
    for unit_num in df['unit_number'].unique():
        max_life = df['time'].loc[df['unit_number'] == unit_num].max()
        lifes[unit_num] = max_life
    # While max life of test data are determined by adding RUL in df_RUL
    if df_RUL is not None:
        for i, rul in df_RUL.itertuples():
            lifes[i + 1] += rul

    # RUL = max_life - time_now
    return df['unit_number'].apply(lambda n: lifes[n]) - df['time']


def load_data(input_dir: str, col_names: List[str]):
    """ Load CMAPSS data from .txt files """
    loaded = {}
    for i in range(4):
        dataset = 'FD_00{}'.format(i + 1)
        f = os.path.join(input_dir, 'train_FD00{}.txt'.format(i + 1))
        df_train = pd.read_csv(f, sep=' ', names=col_names)
        f = os.path.join(input_dir, 'test_FD00{}.txt'.format(i + 1))
        df_test = pd.read_csv(f, sep=' ', names=col_names)
        f = os.path.join(input_dir, 'RUL_FD00{}.txt'.format(i + 1))
        df_RUL = pd.read_csv(f, names=['RUL'])

        loaded[dataset] = {'df_train': df_train,
                           'df_test': df_test,
                           'df_RUL': df_RUL}

    print('Datasets {} are loaded successfully!'.format(
          ', '.join(loaded.keys())))
    return loaded


def one_hot_encode(labels, n_classes):
    """
    Reformat and reshape the input data and labels into TensorFlow format.

    Args:
        labels (array): input labels
        n_classes (int): Number of classes
    Returns:
        labels(ndarray): one-hot-encoded labels
    """
    # Map 0 to (1.0, 0.0, 0.0 ...), 1 to (0.0, 1.0, 0.0 ...)
    labels = (np.arange(n_classes) == labels[:, None]).astype(np.float32)
    return labels


def plot_grouped_by_RUL(df, custom_plot, cols_data: str = None, height: int = 2,
                        leg: bool = True, alpha: float = 0.05,
                        markersize: int = None):
    """ """
    if cols_data is None:
        cols = [col for col in df.columns if len(df[col].unique()) > 2]
        cols_data = [col for col in cols
                     if col.startswith('sen') or col.startswith('os')]

    color = {u: 'grey' for u in df['unit_number'].unique()}
    g = sns.PairGrid(data=df, x_vars="RUL", y_vars=cols_data, palette=color,
                     hue="unit_number", height=height, aspect=6)
    g = g.map(custom_plot, markersize=markersize, alpha=alpha)
    g = g.set(xlim=(df['RUL'].max(), df['RUL'].min() - 10))
    if leg:
        g = g.add_legend()


def to_array(df: pd.DataFrame, window_size: int, output_df: bool = False):
    """
    Transform dataframes into arrays time series sequences.

    Args:
        df (pd.DataFrame): Input data in data frame.
        window_size (int): Window size or the sequence length of the output
                           time series data.
        output_df (bool, optional): Set True to return time series in data frame.

    Returns:
        x (np.array): Array of time series sensor data.
        y (np.array): Array of RULs.
        df_out (pd.DataFrame, optional): Data frame of the time series data.
    """
    concat_dfs = []
    concat_arrays = []
    unit_grps = df.groupby('unit_number')
    for _, unit in unit_grps:
        # create time series array
        # ignore columns of 'unit_number' and 'time'
        array = unit.copy().values[:, 2:]
        num_of_rows = array.shape[0] - window_size + 1
        # sliding window indexer
        indexes = (np.expand_dims(np.arange(window_size), 0) + \
                   np.expand_dims(np.arange(num_of_rows), 0).T)
        array = array[indexes]
        concat_arrays.append(array)
        # testing the validity of the time series arrays
        # for j in range(1, len(array)):
        #     compare = array[j][:window_size - 1] == array[j - 1][1:]
        #     assert compare.all(), 'Error in time series array'
        # create dataframe corresponding to the time series array
        if output_df:
            unit_df = unit.iloc[window_size - 1:][['RUL', 'unit_number']]
            concat_dfs.append(unit_df)

    array = np.concatenate(concat_arrays)
    # slice the array into independent and dependent data
    x, y = array[:, :, :-1], array[:, :, -1]
    y = np.min(y, axis=-1)
    if output_df:
        df_out = pd.concat(concat_dfs, ignore_index=True)
        return x, y, df_out

    return x, y
