import os
import sys
import keras
import numpy as np
from typing import List
from keras import models
from keras import layers
from keras import backend as K
from datetime import datetime
from sklearn.metrics import classification_report
sys.path.append('srcs')
from keras_models import custom_layers


class AbstractModel(object):
    """ Abstract predictive maintenance model """
    def __init__(self, task: str, args: dict = None, load_from: str = None):
        error_mssg = "Argument task only accept 'classification' or " \
                     "'regression', '{}' is given.".format(task)
        assert task in ['classification', 'regression'], error_mssg
        self.args = args
        self.task = task
        self.__model = self._build_model(load_from)

    def train(self,
              train: List[np.ndarray],
              val: List[np.ndarray],
              batch_size: int = 64,
              epochs: int = 100,
              eval_per_epoch: int = 5):
        """ """
        self.__model.summary()
        for i in range(epochs // eval_per_epoch):
            self.__model.fit(train[0], train[1], batch_size=batch_size,
                             epochs=(i + 1) * eval_per_epoch, verbose=1,
                             initial_epoch=i * eval_per_epoch, shuffle=True)
            print('Validation')
            loss, acc = self.__model.evaluate(val[0], val[1],
                                              batch_size=batch_size)
            print('Total iter: {} - val_loss: {:.4f} - val_acc: {:.4f}'.format(
                (i + 1) * eval_per_epoch, loss, acc,
            ))

        # save model
        name = '{}_{}{:02d}{:02d}{:02d}{:02d}.h5'.format(
            self.task, datetime.now().year, datetime.now().month,
            datetime.now().day, datetime.now().hour, datetime.now().minute,
        )
        os.makedirs(os.path.join('models', self._model_name), exist_ok=True)
        self.__model.save(os.path.join('models', self._model_name, name))

    def test(self, test_x, test_y):
        """ """
        print('\nEvaluating...')
        loss, acc = self.__model.evaluate(test_x, test_y, batch_size=256)
        print('val_loss: {:.4f} - val_acc: {:.4f}'.format(loss, acc,))
        return loss, acc

    def predict(self, test_x):
        """ """
        # pred = self.__model.


class AttentionModel(AbstractModel):
    """ Attention based predictive maintenance model """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_name = 'attention'  # name for saving model checkpoint

    def _build_model(self, load_from):
        args = self.args
        if load_from is None:
            # instantiate model
            input_seq = layers.Input(
                shape=(args['window_size'], args['feature_dim']),
                name='input_sequence',
            )
            position_emb = custom_layers.PositionEmbedding()(input_seq)
            # hidden layers
            attn_out = custom_layers.attention_blocks(position_emb, args)
            if args['global_pool'] == 'max':
                attn_out = layers.GlobalMaxPooling1D()(attn_out)
            elif args['global_pool'] == 'mean':
                attn_out = layers.GlobalAveragePooling1D()(attn_out)
            else:
                attn_out = layers.Flatten()(attn_out)

            attn_out = layers.Dropout(0.15)(attn_out)
            # output layer
            out = custom_layers.fc_net(attn_out, args)
            # apply softmax for classification task
            if self.task == 'classification':
                out = layers.Activation('softmax')(out)
            else:
                out = layers.Activation('linear')(out)

            # model compilation
            loss_func = 'mse' * (self.task == 'regression') +\
                        'binary_crossentropy' * (self.task == 'classification')
            metrics = 'mae' * (self.task == 'regression') +\
                      'accuracy' * (self.task == 'classification')
            model = models.Model(input_seq, out)
            model.compile(optimizer='adam', metrics=[metrics], loss=loss_func)
        else:
            custom_objects = {
                'PositionEmbedding': custom_layers.PositionEmbedding,
                'LayerNormalization': custom_layers.LayerNormalization,
                'FeedForward': custom_layers.FeedForward,
                'AttentionLayer': custom_layers.AttentionLayer,
            }
            model = models.load_model(load_from,
                                      custom_objects=custom_objects)

        return model


class CNNAttentionModel(AbstractModel):
    """ CNN-Attention predictive maintenance model """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_name = 'cnn_attention'  # name for saving model checkpoint

    def _build_model(self, load_from):
        args = self.args
        if load_from is None:
            # instantiate model
            input_seq = layers.Input(
                shape=(args['window_size'], args['feature_dim']),
                name='input_sequence',
            )
            conv = layers.Conv1D(128, 3, activation='relu', padding='same')(input_seq)
            conv = layers.BatchNormalization()(conv)
            conv = layers.MaxPooling1D(2)(conv)
            position_emb = custom_layers.PositionEmbedding()(conv)
            # hidden layers
            attn_out = custom_layers.attention_blocks(position_emb, args)
            if args['global_pool'] == 'max':
                attn_out = layers.GlobalMaxPooling1D()(attn_out)
            elif args['global_pool'] == 'mean':
                attn_out = layers.GlobalAveragePooling1D()(attn_out)
            else:
                attn_out = layers.Flatten()(attn_out)
            attn_out = layers.Dropout(0.15)(attn_out)
            # attn_out = layers.LSTM(100)(conv)
            # output layer
            out = custom_layers.fc_net(attn_out, args)
            # apply softmax for classification task
            if self.task == 'classification':
                out = layers.Activation('softmax')(out)
            else:
                out = layers.Activation('linear')(out)

            # model compilation
            loss_func = 'mse' * (self.task == 'regression') +\
                        'binary_crossentropy' * (self.task == 'classification')
            metrics = 'mae' * (self.task == 'regression') +\
                      'accuracy' * (self.task == 'classification')
            model = models.Model(input_seq, out)
            model.compile(optimizer='adam', metrics=[metrics], loss=loss_func)
        else:
            custom_objects = {
                'PositionEmbedding': custom_layers.PositionEmbedding,
                'LayerNormalization': custom_layers.LayerNormalization,
                'FeedForward': custom_layers.FeedForward,
                'AttentionLayer': custom_layers.AttentionLayer,
            }
            model = models.load_model(load_from,
                                      custom_objects=custom_objects)
        return model


class CNNLSTMModel(AbstractModel):
    """ CNN-Attention predictive maintenance model """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_name = 'cnn_attention'  # name for saving model checkpoint

    def _build_model(self, load_from):
        args = self.args
        if load_from is None:
            # instantiate model
            input_seq = layers.Input(
                shape=(args['window_size'], args['feature_dim']),
                name='input_sequence',
            )
            conv = layers.Conv1D(128, 3, activation='relu', padding='same')(input_seq)
            conv = layers.BatchNormalization()(conv)
            conv = layers.MaxPooling1D(2)(conv)
            lstm_out = layers.LSTM(100)(conv)
            # output layer
            out = custom_layers.fc_net(lstm_out, args)
            # apply softmax for classification task
            if self.task == 'classification':
                out = layers.Activation('softmax')(out)
            else:
                out = layers.Activation('linear')(out)

            # model compilation
            loss_func = 'mse' * (self.task == 'regression') +\
                        'binary_crossentropy' * (self.task == 'classification')
            metrics = 'mae' * (self.task == 'regression') +\
                      'accuracy' * (self.task == 'classification')
            model = models.Model(input_seq, out)
            model.compile(optimizer='adam', metrics=[metrics], loss=loss_func)
        else:
            model = models.load_model(load_from)
        return model
