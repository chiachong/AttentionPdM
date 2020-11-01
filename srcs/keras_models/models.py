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


class AttentionModel(object):
    """ """
    def __init__(self, task: str, args: dict = None, load_from: str = None):
        error_mssg = "Argument task only accept 'classification' or " \
                     "'regression', '{}' is given.".format(task)
        assert task in ['classification', 'regression'], error_mssg
        self.args = args
        self.load_from = load_from
        self.task = task
        self.__model = None

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
            # output layer
            out = custom_layers.fc_net(attn_out, args)
            # apply softmax for classification task
            if self.task == 'classification':
                out = layers.Activation('softmax')(out)
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
            model = models.load_model(self.load_model,
                                      custom_objects=custom_objects)
        return model

    def train(self,
              train: List[np.ndarray],
              val: List[np.ndarray],
              batch_size: int = 64,
              epochs: int = 100,
              eval_per_epoch: int = 5):
        """ """
        if self.__model is None:
            self.__model = self._build_model()

        self.__model.summary()
        for i in range(epochs // eval_per_epoch):
            self.__model.fit(train[0], train[1], batch_size=batch_size,
                             epochs=(i + 1) * eval_per_epoch, verbose=1,
                             initial_epoch=i * eval_per_epoch)
            print('Validation')
            loss, acc = self.__model.evaluate(val[0], val[1],
                                              batch_size=batch_size)
            print('Total iter: {} - val_loss: {:.4f} - val_acc: {:.4f}'.format(
                (i + 1) * eval_per_epoch, loss, acc,
            ))

        # save model
        folder = 'attention_classification_{}{:02d}{:02d}{:02d}{:02d}'.format(
            datetime.now().year, datetime.now().month, datetime.now().day,
            datetime.now().hour, datetime.now().minute,
        )
        os.makedirs(os.path.join('models', folder))
        self.__model.save(os.path.join('models', folder, 'model.h5'))

    def test(self):
        """ """
        pass

    def predict(self):
        """ """
        pass
