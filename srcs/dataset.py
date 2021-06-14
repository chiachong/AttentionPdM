import sys
import numpy as np
import pandas as pd
from typing import List
sys.path.append('srcs')
import utils

# list of column names in the NASA turbofan data
COL_NAMES = ['unit_number', 'time', 'os_1', 'os_2', 'os_3']
COL_NAMES += ['sensor_{}'.format(s + 1) for s in range(23)]
# list of columns to drop
DROP_COLS = ['sensor_16', 'sensor_19', 'sensor_22', 'sensor_23']


class TurbofanData(object):
    """ Class object for NASA Turbofan Engine data """
    def __init__(self, input_dir: str):
        self.data = utils.load_data(input_dir, COL_NAMES)

    def __getitem__(self, dataset):
        return self.data[dataset]

    def _normalize(self, drop_cols: List[str]):
        """ Normalize the data using min-max scaler """
        # columns that will be normalized
        cols = [c for c in COL_NAMES
                if c not in ['unit_number', 'time', 'RUL'] + drop_cols]
        # first step is concat the 4 train df
        to_concat = []
        for i in range(4):
            dataset = 'FD_00{}'.format(i + 1)
            df = self.data[dataset]['df_train'][cols].copy()
            to_concat.append(df)
            # df = self.data[dataset]['df_test'][cols].copy()
            # to_concat.append(df)

        concated_df = pd.concat(to_concat)
        # second step is to calculate the min and max values in each columns
        # min max values are calculated using train data only
        xmin = concated_df.values.min(axis=0)
        xmax = concated_df.values.max(axis=0)
        # third step is apply the normalization on both train and test sets
        for i in range(4):
            dataset = 'FD_00{}'.format(i + 1)
            # normalize train df
            df = self.data[dataset]['df_train'][cols].copy()
            normalized = (df - xmin) / (xmax - xmin)
            self.data[dataset]['df_train'][cols] = normalized
            # normalize test df
            df = self.data[dataset]['df_test'][cols].copy()
            normalized = (df - xmin) / (xmax - xmin)
            self.data[dataset]['df_test'][cols] = normalized

    def _to_array(self, sets: str, window_size: int) -> np.ndarray:
        """ Transform dataframes into arrays """
        to_concat_x = []
        to_concat_y = []
        for i in range(4):
            dataset = 'FD_00{}'.format(i + 1)
            df = self.data[dataset]['df_' + sets]
            x, y = utils.to_array(df, window_size)
            to_concat_x.append(x)
            to_concat_y.append(y)

        x = np.concatenate(to_concat_x)
        y = np.concatenate(to_concat_y)
        return x, y

    def preprocess(self, drop_cols: List[str] = DROP_COLS,
                   normalize: bool = True, clip_RUL: int = None,
                   moving_averages: List[int] = []):
        """
        Preprocess the loaded dataframes

        Args:
            drop_cols (List[str], optional):
                Columns to drop.
                Defaults to ['sensor_16', 'sensor_19', 'sensor_22', 'sensor_23']
            normalize (bool, optional):
                Set True to normalize the data using Min-Max scaler.
                Defaults to True.
            clip_RUL (int, optional):
                Maximum value of RUL to clip. Defaults to None.
            moving_averages (List[int], optional):
                List of window sizes for moving average.
                Defaults to [].
        """
        for i in range(4):
            dataset = 'FD_00{}'.format(i + 1)
            # drop columns
            if drop_cols:
                self.data[dataset]['df_train'].drop(columns=drop_cols,
                                                    inplace=True)
                self.data[dataset]['df_test'].drop(columns=drop_cols,
                                                   inplace=True)

            # calculate RUL
            df_train = self.data[dataset]['df_train']
            df_test = self.data[dataset]['df_test']
            df_RUL = self.data[dataset]['df_RUL']
            train_RUL = utils.calculate_RUL(df_train)
            test_RUL = utils.calculate_RUL(df_test, df_RUL)
            # clip the maximum RUL to a certain value
            if clip_RUL is not None:
                train_RUL = np.clip(train_RUL, None, clip_RUL)
                test_RUL = np.clip(test_RUL, None, clip_RUL)

            # add RUL to the dataframes
            self.data[dataset]['df_train']['RUL'] = train_RUL
            self.data[dataset]['df_test']['RUL'] = test_RUL
            print('Finish calculating RUL in {}.'.format(dataset))

        # moving averages
        moving_averages = list(set(moving_averages))  # deduplicate window sizes
        if len(moving_averages) > 0:
            col_names = [c for c in COL_NAMES
                         if c not in drop_cols + ['unit_number', 'time', 'RUL']]
            for train_test in ['train', 'test']:
                for i in range(4):
                    dataset = 'FD_00{}'.format(i + 1)
                    df = self.data[dataset][f'df_{train_test}'].copy()
                    new_cols = {}
                    for w_size in moving_averages:
                        new = utils.calculate_moving_average(df, w_size, col_names)
                        new_cols.update(new)
                    # add calculated moving averages into df
                    for new_name, new_col in new_cols.items():
                        df[new_name] = new_col
                    # move the RUL column to the last column
                    cols = df.columns.tolist()
                    rul_index = cols.index('RUL')
                    cols = cols[:rul_index] + cols[rul_index + 1:] + ['RUL']
                    df = df[cols]
                    # remove na then update into self.data
                    self.data[dataset][f'df_{train_test}'] = df.dropna()

            print('Finish calculating moving averages.')

        # normalize train and test data sets
        if normalize:
            self._normalize(drop_cols)
            print('Finish normalizing train and test sets.')

    def split_train_val(self, train_p: float, seed: int = 1234):
        """
        Split data into training and validation sets by randomly pick
        train_p proportion of unit numbers to be training set, the rest
        would be validation set.

        Args:
            train_p: Proportion of training data.
            seed: Random seed number.
        """
        error_mssg = 'Data is not preprocessed! Please run .preprocess() '\
                     'method before split train val!'
        assert 'RUL' in self.data['FD_001']['df_train'].columns, error_mssg
        np.random.seed(seed)  # set seed
        for i in range(4):
            dataset = 'FD_00{}'.format(i + 1)
            df = self.data[dataset]['df_train'].copy()
            total_unit = df['unit_number'].unique()
            # pick the unit numbers for training set
            train_unit = np.random.choice(total_unit,
                                          int(train_p * len(total_unit)),
                                          replace=False)
            # split training set
            df_train = df.loc[
                df['unit_number'].apply(lambda x: x in train_unit)
            ].copy()
            # split validation set
            df_val = df.loc[
                df['unit_number'].apply(lambda x: x not in train_unit)
            ].copy()
            self.data[dataset]['df_val'] = df_val
            self.data[dataset]['df_train'] = df_train
            print('{}: {} train, {} validation'.format(
                dataset, len(train_unit), len(total_unit) - len(train_unit)
            ))

    def arrays_for_classification(self,
                                  window_size: int,
                                  fail_in: int) -> np.array:
        """
        This will return two np.array which are independent variables and
        dependent variables. The dependent variables are boolean type,
        such that if the min(RUL) in the input window size <= fail_in,
        return True else False.

        Args:
            window_size: Size of the scanning window.
            fail_in: Failure occur within the specified number of operations.

        Returns:
            train_x: Dependent variables in the training data,
                     shape=(num_example, window, num_col)
            train_y: Independent variable in the training data,
                     shape=(num_example, )
            test_x: Dependent variables in the teting data,
                    shape=(num_example, window, num_col)
            test_y: Independent variable in the teting data,
                    shape=(num_example, )
            val_x: Dependent variables in the validation data,
                   shape=(num_example, window, num_col)
            val_y: Independent variable in the validation data,
                   shape=(num_example, )
        """
        train_x, train_y = self._to_array('train', window_size)
        train_y = np.where(train_y <= fail_in, 1, 0)
        test_x, test_y = self._to_array('test', window_size)
        test_y = np.where(test_y <= fail_in, 1, 0)
        val_x, val_y = self._to_array('val', window_size)
        val_y = np.where(val_y <= fail_in, 1, 0)
        return train_x, train_y, test_x, test_y, val_x, val_y

    def arrays_for_regression(self,
                              window_size: int) -> np.array:
        """
        This will return two np.array which are independent variables and
        dependent variables. The dependent variables are the last RUL of the
        respective time series.

        Args:
            window_size: Size of the scanning window.

        Returns:
            train_x: Dependent variables in the training data,
                     shape=(num_example, window, num_col)
            train_y: Independent variable in the training data,
                     shape=(num_example, )
            test_x: Dependent variables in the teting data,
                    shape=(num_example, window, num_col)
            test_y: Independent variable in the teting data,
                    shape=(num_example, )
            val_x: Dependent variables in the validation data,
                   shape=(num_example, window, num_col)
            val_y: Independent variable in the validation data,
                   shape=(num_example, )
        """
        train_x, train_y = self._to_array('train', window_size)
        test_x, test_y = self._to_array('test', window_size)
        val_x, val_y = self._to_array('val', window_size)
        return train_x, train_y, test_x, test_y, val_x, val_y
