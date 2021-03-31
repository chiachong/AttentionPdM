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
        self.load_from = load_from
        self.task = task
        self._model = None

    def train(self,
              train: List[np.ndarray],
              val: List[np.ndarray],
              batch_size: int = 64,
              epochs: int = 100,
              eval_per_epoch: int = 5,
              learning_rate_decay: float = None):
        """ """
        if self._model is None:
            self._build_model()

        self._model.summary()
        for i in range(epochs // eval_per_epoch):
            self._model.fit(train[0], train[1], batch_size=batch_size,
                            epochs=(i + 1) * eval_per_epoch, verbose=1,
                            initial_epoch=i * eval_per_epoch, shuffle=True)
            print('Validation')
            loss, acc = self._model.evaluate(val[0], val[1],
                                             batch_size=batch_size)
            print('Total iter: {} - val_loss: {:.4f} - val_acc: {:.4f}'.format(
                (i + 1) * eval_per_epoch, loss, acc,
            ))
            if learning_rate_decay is not None:
                _lr = K.get_value(self._model.optimizer.lr)
                new_lr = _lr * (1 - learning_rate_decay)
                K.set_value(self._model.optimizer.lr, new_lr)

        # save model
        name = '{}_{}{:02d}{:02d}{:02d}{:02d}.h5'.format(
            self.task, datetime.now().year, datetime.now().month,
            datetime.now().day, datetime.now().hour, datetime.now().minute,
        )
        os.makedirs(os.path.join('models', self._model_name), exist_ok=True)
        self._model.save(os.path.join('models', self._model_name, name))

    def test(self, test_x, test_y):
        """ """
        if self._model is None:
            self._build_model()

        print('\nEvaluating...')
        if self.task == 'regression':
            loss, mae = self._model.evaluate(test_x, test_y, batch_size=256)
            print('val_loss: {:.4f} - val_mae: {:.4f}'.format(loss, mae))
            return loss, mae
        else:
            pred = self.predict(test_x)
            test_y = np.argmax(test_y, axis=-1)
            target_names = ['Functioning', 'Fail']
            print(classification_report(test_y, pred, target_names=target_names))
            report = classification_report(test_y, pred, output_dict=True)['1']
            return report['precision'], report['recall'], report['f1-score']

    def predict(self, test_x):
        """ """
        if self._model is None:
            self._build_model()

        pred = self._model.predict(test_x)
        pred = np.argmax(pred, axis=-1)
        return pred


class AttentionModel(AbstractModel):
    """ Attention based predictive maintenance model """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_name = 'attention'  # name for saving model checkpoint

    def _build_model(self):
        args = self.args
        if self.load_from is None:
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
            model = models.load_model(self.load_from,
                                      custom_objects=custom_objects)

        self._model = model


class CNNAttentionModel(AbstractModel):
    """ CNN-Attention predictive maintenance model """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_name = 'cnn_attention'  # name for saving model checkpoint

    def _build_model(self):
        args = self.args
        if self.load_from is None:
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
            model = models.load_model(self.load_from,
                                      custom_objects=custom_objects)

        self._model = model


class CNNLSTMModel(AbstractModel):
    """ CNN-Attention predictive maintenance model """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_name = 'cnn_lstm'  # name for saving model checkpoint

    def _build_model(self):
        args = self.args
        if self.load_from is None:
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
            model = models.load_model(self.load_from)

        self._model = model
