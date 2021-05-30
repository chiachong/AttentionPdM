# AttentionPdM
Predictive Maintenance Using Attention Model.

## Prepare Environments
The codes were tested and ran on Ubuntu 18.04 using python 3.7.5. 
Create and set up a python environment by running the following command in the terminal
```
# install the gpu version if there is any compatible NVIDIA GPU
source ./create_env_gpu

# else install the cpu version
source ./create_env_cpu
```
then the environment can be activated by
```
source ./activate
```
It is important to activate the environment before carry out any process discussed below.

## Experiment Records
The experiment records were recorded using [MLFlow](https://mlflow.org/), which can be accessed by running the command in the terminal
```
mlflow ui
```

## Jupyter Notebooks
There are some notebooks in this [repo](/srcs/notebooks) for data exploratory analysis. They can be accessed by running the following command in the root of this repo
```
jupyter notebook
```

## Experiment Data
The publicly available [data](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/): NASA's turbodan engine degradation simulation data set was used in the experiments. This dataset consists of four sub-datasets with different operation settings and fault conditions. Each sub-dataset consists of simulated aero turbofan engines run-to-failure data with each engine having 24 sensors. The raw data is already provided in this repo and can be accessed by
```python
from srcs import dataset

data_dir = 'data/CMAPSSData'
data = dataset.TurbofanData(data_dir)

# preprocess data
data.preprocess(drop_cols=['sensor_16', 'sensor_19', 'sensor_22', 'sensor_23'],  # list of column names to drop
                normalize=True,  # set True to normalize data using Min-Max scaler
                clip_RUL=100)  # clip the maximum of the RUL to 100

# split training and validation data sets
data.split_train_val(train_p=3 / 4)  # randomly pick 3/4 of the unit numbers to be training set,
                                     # the rest would be validation set

# generate time series arrays for model training and evaluation
arrays = data.arrays_for_regression(window_size=30)
train_x, train_y, test_x, test_y, val_x, val_y = arrays
# *_x are the sequences of multivariate time series sensor readings with shape (N*, 30, 22)
# *_y are the sequences of RULs with shape (N*, )
# where N* is the number of examples in the training, testing or validation sets.
```

## Types of Model
There are a several models provided in this project, for example:
* AttentionModel
* CuDNNGRUModel
* GRUModel

Note that every model requires a dictionary of condigurations in order to instantiate a model instance.
```python
args = {
    'window_size': 30,  # size of the scanning window
    'feature_dim': 22,  # number of sensors being used
    'hidden_layers': ...,  # will be discussed in model explaination
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

## How to Train a Model
