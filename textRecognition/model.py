from keras.models import Model
from keras.layers import Conv2D, Bidirectional, LSTM, GRU, Dense
from keras.layers import Dropout, BatchNormalization, LeakyReLU, PReLU, Multiply
from keras.layers import Input, Add, Activation, Lambda, MaxPooling2D, Reshape
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop
from keras.constraints import MaxNorm

import os
import keras
import numpy as np
import tensorflow as tf
import keras.backend as K

# INPUT_SIZE = (256, 64, 1)
# MAX_WORD_LENGTH = 64
# CHARSET = WordHTRFlor.LATIN_CHAR

class FullGatedConv2D(Conv2D):
    """Gated Convolutional Class"""

    def __init__(self, filters, **kwargs):
        super(FullGatedConv2D, self).__init__(filters=filters * 2, **kwargs)
        self.nb_filters = filters

    def call(self, inputs):
        """Apply gated convolution"""
        output = super(FullGatedConv2D, self).call(inputs)
        linear = Activation("linear")(output[:, :, :, :self.nb_filters])
        sigmoid = Activation("sigmoid")(output[:, :, :, self.nb_filters:])

        return Multiply()([linear, sigmoid])

    def compute_output_shape(self, input_shape):
        """Compute shape of layer output"""
        output_shape = super(FullGatedConv2D, self).compute_output_shape(input_shape)
        return tuple(output_shape[:3]) + (self.nb_filters,)

    def get_config(self):
        """Return the config of the layer"""
        config = super(FullGatedConv2D, self).get_config()
        config['nb_filters'] = self.nb_filters
        del config['filters']
        return config


class WordHTRFlor:

    ASCII_CHAR = " !\"#$%&'()*+,-.0123456789:;<>@ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    LATIN_CHAR = " !\"#$%&'()*+,-.0123456789:;<>@ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzáÁéÉíÍóÓúÚëËïÏüÜñÑçÇâÂêÊîÎôÔûÛàÀèÈùÙ"
    DECODER_CONFIG = { "greedy": False, "beam_width": 10, "top_paths": 1 }

    def __init__(self, logdir, input_size=(256, 64, 1), charset=None, optimizer=None, decoder_conf=None, verbose=0):
        self.charset = charset or WordHTRFlor.LATIN_CHAR
        self.model_outputs = len(self.charset) + 1
        self.logdir = logdir

        self.callbacks = self._set_callbacks(verbose, monitor="val_loss")
        self.model = self._build_model(input_size, optimizer)
        self.decoder_conf = decoder_conf or WordHTRFlor.DECODER_CONFIG

    def load_chkpt(self, fpath):
        """ Load a model with checkpoint file"""
        self.model.load_weights(fpath)

    def _set_callbacks(self, verbose, monitor):
        """Setup the list of callbacks for the model"""
        callbacks = [
            CSVLogger(filename=os.path.join(self.logdir, "epochs.log"), separator=";", append=True),
            TensorBoard(
                log_dir=self.logdir,
                histogram_freq=10,
                profile_batch=0,
                write_graph=True,
                write_images=False,
                update_freq="epoch"),
            ModelCheckpoint(
                filepath=f"{self.logdir}/checkpoint.params",
                monitor=monitor,
                save_best_only=True,
                save_weights_only=True,
                verbose=verbose),
            EarlyStopping(
                monitor=monitor,
                min_delta=1e-8,
                patience=20,
                restore_best_weights=True,
                verbose=verbose),
            ReduceLROnPlateau(
                monitor=monitor,
                min_delta=1e-8,
                factor=0.2,
                patience=15,
                verbose=verbose)
        ]

        return callbacks
    
    @staticmethod
    def ConvLayer(input_layer, filters, kernels, strides, add_dropout=False, add_fullgconv=False):
        cnn = Conv2D(
            filters=filters,
            kernel_size=kernels[0],
            strides=strides,
            padding="same",
            kernel_initializer="he_uniform"
        )(input_layer)
        cnn = PReLU(shared_axes=[1,2])(cnn)
        cnn = BatchNormalization()(cnn)

        if add_fullgconv:
            cnn = FullGatedConv2D(
                filters=filters,
                kernel_size=kernels[-1],
                padding="same",
                kernel_constraint=MaxNorm(4, [0,1,2])
            )(cnn)

            if add_dropout:  # only add it when adding a FullGatedConv layer previously
                cnn = Dropout(rate=0.2)(cnn)

        return cnn
        
    @staticmethod
    def ctc_loss_lambda_func(y_true, y_pred):
        """Function for computing the CTC loss"""
        if len(y_true.shape) > 2:
            y_true = tf.squeeze(y_true)

        # y_pred.shape = (batch_size, string_length, alphabet_size_1_hot_encoded)
        # output of every model is softmax
        # so sum across alphabet_size_1_hot_encoded give 1
        #               string_length give string length
        input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
        input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)

        # y_true strings are padded with 0
        # so sum of non-zero gives number of characters in this string
        label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")

        loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

        # average loss across all entries in the batch
        loss = tf.reduce_mean(loss)

        return loss

    def _build_model(self, input_size, optimizer=None):
        """Configures the HTR Model for training/predict. Uses Flor model.
        Source:
            https://github.com/arthurflor23/handwritten-text-recognition
        """

        input_data = Input(name="input", shape=input_size)

        cnn = WordHTRFlor.ConvLayer(input_data, 16, [(3,3), (3,3)], (2,2), add_dropout=False, add_fullgconv=True)

        cnn = WordHTRFlor.ConvLayer(cnn, 32, [(3,3), (3,3)], (1,1), add_dropout=False, add_fullgconv=True)

        cnn = WordHTRFlor.ConvLayer(cnn, 40, [(2,4), (3,3)], (2,4), add_dropout=True, add_fullgconv=True)
        cnn = WordHTRFlor.ConvLayer(cnn, 48, [(3,3), (3,3)], (1,1), add_dropout=True, add_fullgconv=True)
        cnn = WordHTRFlor.ConvLayer(cnn, 56, [(2,4), (3,3)], (2,4), add_dropout=True, add_fullgconv=True)

        cnn = WordHTRFlor.ConvLayer(cnn, 64, [(3,3), (None, None)], (1,1), add_dropout=False, add_fullgconv=False)

        cnn = MaxPooling2D(pool_size=(1,2), strides=(1,2), padding="valid")(cnn)

        shape = cnn.get_shape()
        nb_units = shape[2] * shape[3]

        bgru = Reshape((shape[1], nb_units))(cnn)

        bgru = Bidirectional(GRU(units=nb_units, return_sequences=True, dropout=0.5))(bgru)
        bgru = Dense(units=nb_units * 2)(bgru)

        bgru = Bidirectional(GRU(units=nb_units, return_sequences=True, dropout=0.5))(bgru)
        output_data = Dense(units=self.model_outputs, activation="softmax")(bgru)


        net_optimizer = optimizer or RMSprop(learning_rate=5e-4)
        model = Model(inputs=input_data, outputs=output_data)
        model.compile(optimizer=net_optimizer, loss=WordHTRFlor.ctc_loss_lambda_func)
        return model

    def predict(self, x, batch_size=None, verbose=0, steps=1, callbacks=None, max_queue_size=10, workers=1,
        use_multiprocessing=False, ctc_decode=True):
        """
        Model predicting on data yielded (predict function has support to generator).
        A predict() abstration function of TensorFlow 2.

        Provide x parameter of the form: yielding [x].

        :param: See tensorflow.keras.Model.predict()
        :return: raw data on `ctc_decode=False` or CTC decode on `ctc_decode=True` (both with probabilities)
        """

        self.model._make_predict_function()

        if verbose == 1:
            print("Model Predict")

        out = self.model.predict(
            x=x, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks, max_queue_size=max_queue_size,
            workers=workers, use_multiprocessing=use_multiprocessing
        )

        if not ctc_decode:
            return np.log(out.clip(min=1e-8))

        steps_done = 0
        if verbose == 1:
            print("CTC Decode")
            progbar = tf.keras.utils.Progbar(target=steps)

        batch_size = int(np.ceil(len(out) / steps))
        input_length = len(max(out, key=len))

        predicts, probabilities = [], []
        dconf1 = self.decoder_conf["greedy"]
        dconf2 = self.decoder_conf["beam_width"]
        dconf3 = self.decoder_conf["top_paths"]

        while steps_done < steps:
            index = steps_done * batch_size
            until = index + batch_size

            x_test = np.asarray(out[index:until])
            x_test_len = np.asarray([input_length for _ in range(len(x_test))])

            decode, log = K.ctc_decode(
                x_test, x_test_len, greedy=dconf1, beam_width=dconf2, top_paths=dconf3
            )

            probabilities.extend([np.exp(x) for x in log])
            decode = [[[int(p) for p in x if p != -1] for x in y] for y in decode]
            predicts.extend(np.swapaxes(decode, 0, 1))

            steps_done += 1
            if verbose == 1:
                progbar.update(steps_done)

        return (predicts, probabilities)