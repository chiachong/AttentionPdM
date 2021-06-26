# AttentionPdM
The ability to predict the remaining useful life (RUL) of a machine plays an important role in the safety and competency of the industrial system. 
In this project, a deep learning model based on the attention mechanism (Vaswani et al., 2017) was developed to predict the RUL based on the time series machine degradation data. 

## Table of Contents
* [Prepare Environments](#prepare-environments)
* [About Data](#about-data)
* [Jupyter Notebooks](#jupyter-notebooks)
* [Experiment Results](#experiment-results)
* [Types of Model](#types-of-model)
* [How to Train a Model](#how-to-train-a-model)
* [How to Test a Model](#how-to-test-a-model)
* [References](#references)

## Prepare Environments
The codes were tested and ran on Ubuntu 18.04 using python 3.7.5. 
Create and set up a python environment by running the following command in the terminal
```
# install the gpu version if there is compatible NVIDIA GPU
source ./create_env_gpu

# else install the cpu version
source ./create_env_cpu
```
then the environment can be activated by
```
source ./activate
```
It is important to activate the environment before carry out any process discussed below.

## About Data
The publicly available [data](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/):
 NASA's turbodan engine degradation simulation data set was used in the experiments. 
This dataset consists of four sub-datasets with different operation settings and fault conditions. 
Each sub-dataset consists of simulated aero turbofan engines run-to-failure data with each engine having 24 sensors. 
The raw data is already provided in this [repo](/data/CMAPSSData) and can be accessed by
```python
from srcs import dataset

data_dir = 'data/CMAPSSData'
data = dataset.TurbofanData(data_dir)

# to get the raw dataframes of a sub-dataset
# generally, df = data['FD_00x']['df_train']
train_df1 = data['FD_001']['df_train']  # dataframe from the train_FD001.txt
test_df2 = data['FD_002']['df_test']  # dataframe from the test_FD002.txt
rul_df3 = data['FD_003']['df_RUL']  # dataframe from the RUL_FD003.txt
train_df4, test_df4, rul_df4 = data['FD_004'].values()  # return the 3 dataframes related to FD004

# preprocess data
data.preprocess(drop_cols=['sensor_16', 'sensor_19', 'sensor_22', 'sensor_23'],  # list of column names to drop
                normalize=True,  # set True to normalize data using Min-Max scaler
                clip_RUL=100)  # clip the maximum of the RUL to 100

# split training and validation data sets
data.split_train_val(train_p=3 / 4)  # randomly pick 3/4 of the unit numbers to be training set,
                                     # the rest would be the validation set

# generate time series arrays for model training and evaluation
arrays = data.arrays_for_regression(window_size=30)
train_x, train_y, test_x, test_y, val_x, val_y = arrays
# *_x are the sequences of multivariate time series sensor readings with shape (N*, 30, 22)
# *_y are the sequences of RULs with shape (N*, )
# where N* is the number of examples in the training, testing or validation sets.
```

## Jupyter Notebooks
There are some notebooks in this [repo](/srcs/notebooks) for data exploratory analysis. 
Open and execute them by running the following terminal command in the root of the repo
```
jupyter notebook
```

## Experiment Results
Performance metrics of the attention models with different number of attention layers:
| Number of Attention layer |  MAE  |  RMSE  |
| :-----------------------: | ----- | ------ |
|             1             | 7.33  | 11.45  |
|             2             | 6.66  | 10.78  |
|             3             | 7.76  | 11.41  |

Performance metrics of the attention models (2 attention layers) with different window sizes:
| Window Size |  MAE  |  RMSE  |
| :---------: | ----- | ------ |
|     15      | 7.68  | 12.43  |
|     30      | 6.66  | 10.78  |
|     45      | 6.63  | 10.69  |

Comparison of the model performances:
|       Method       |  MAE  |  RMSE  |
| :----------------: | ----- | ------ |
| CNN+LSTM [[1]](#1) | 10.20 | 14.30  |
|    GRU [[3]](#3)   | 6.01  | 10.34  |
|      Attention     | 6.66  | 10.78  |

The experiment results were recorded using [MLflow](https://mlflow.org/), which can be accessed by running the command in the terminal
```
mlflow ui
```

## Types of Model
There are several models provided in this project, for example:
* [AttentionModel](#attentionmodel)
* [GRUModel](#grumodel)
* [CuDNNGRUModel](#cudnngrumodel)

Note that every model requires a dictionary of condigurations in order to instantiate a model instance.
```python
args = {
    'window_size': 30,  # size of the scanning window
    'feature_dim': 22,  # number of sensors being used
#     'hidden_layers': ...,  # will be discussed in model explaination
    'fully_connected_layers': [128, 1],  # number of neurons in the dense layers,
                                         # the last number should always be 1
    'global_pool': 'max',  # global pooling, options='max', 'mean', None
    'dropout_rate': 0.1,  # probability for each neoruon to be dropped out 
    'batch_normalization': True,  # set True to enable batch normalization
    'batch_size': 512,  # number of example per batch
    'epochs': 50,  # number of iterations
    'eval_per_epoch': 5,  # number of iterations per evaluation
    'learning_rate_decay': 0.01,  # decay rate for the learning rate in each new iteration
}
```

### AttentionModel
Attention based predictive maintenance model.
```python
from srcs.keras_models import AttentionModel

args = {
    ...,  # use the examples above
    'hidden_layers': [[4, 32, 50], [4, 32, 50]]
    # this means the model consists of two attention layers,
    # where the first layer contains 4 attention heads, size per head=32 and 50 feed forward units
    # and the second layer also contains 4 attention heads, size per head=32 and 50 feed forward units
}
pdm_model = AttentionModel(task='regression', args=args)
```

### GRUModel
GRU based predictive maintenance model.
```python
from srcs.keras_models import GRUModel

args = {
    ...,  # use the examples above
    'hidden_layers': [100]
    # this means the model consists of one GRU layer with 100 hidden units
}
pdm_model = GRUModel(task='regression', args=args)
```

### CuDNNGRUModel
GRU based (GPU parallelized) predictive maintenance model. 
This model uses the same algorithm as the [GRUModel](#grumodel) however this model enables better GPU parralelization. 
Hence a gpu is required for this model, please use the GRUModel if no gpu available.
```python
from srcs.keras_models import CuDNNGRUModel

args = {
    ...,  # use the examples above
    'hidden_layers': [100]
    # this means the model consists of one GRU layer with 100 hidden units
}
pdm_model = CuDNNGRUModel(task='regression', args=args)
```

## How to Train a Model
Assume we already have a dictionary of configurations defined as `args` and we want to train an attention model, then
```python
from srcs import dataset
from srcs.keras_models import AttentionModel

# experiment and model settings
args = {...}

# load and preprocess data
data = dataset.TurbofanData('data/CMAPSSData')
data.preprocess(clip_RUL=100)
data.split_train_val(3 / 4)
arrays = data.arrays_for_regression(args['window_size'])
train_x, train_y, test_x, test_y, val_x, val_y = arrays

# instantiate model instance
pdm_model = AttentionModel(task='regression', args=args)
# start the training process
pdm_model.train(train=[train_x, train_y],
                val=[val_x, val_y],
                batch_size=args['batch_size'],
                epochs=args['epochs'],
                eval_per_epoch=args['eval_per_epoch'],
                learning_rate_decay=args['learning_rate_decay'])
```
Upon finish model training, a model file `regression_xxxxxxxxxxxx.h5` will be saved into the `./models` folder. 
In addition, the script [`train.py`](/srcs/train.py) illustrates the example to train model and record the experiment using MLflow.
```
# run the example training script in the root of repo
python3 srcs/train.py
```

## How to Test a Model
Once a model is trained, we can evaluate the model by picking a model checkpoint then
```python
model_dir = 'models/attention/regression_xxxxxxxxxxxx.h5'  # eg. trained attention model

# load the trained model
pdm_model = AttentionModel(task='regression', load_from=model_dir)

# model evaluation using test data loaded in the previous section
loss, mae, rmse = pdm_model.test(test_x, test_y)
print(f'Test loss: {loss} - Test MAE: {mae} - Test RMSE: {rmse}')
```

## References
<a id="1">[1]</a> 
Pektas, A., & Pektas, E. (2018). A novel scheme for accurate remaining useful life prediction for industrial IoTs by using deep neural network. *International Journal of Artificial Intelligence and Applications (IJAIA)*, *9*(5), 17-25.

<a id="2">[2]</a> 
Saxena, A., & Goebel, K. (2008). *Prognostics Center - Data Repository*. 
Retrieved from National Aeronautics and Space Administration (NASA):
 https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

<a id="3">[3]</a> 
Suursalu, S. (2017). Predictive maintenance using machine learning methods in petrochemical refineries. (Unpublished master's thesis). Delft University of Technology, Delft, Netherlands.

<a id="4">[4]</a> 
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). 
Attention is all you need. 
*31st Conference on Neural Information Processing Systems (NIPS 2017)*, (pp. 6000-6010). Long Beach, CA, USA.
