import datetime
import os
import random as rn
import json
import subprocess
import sys

import tensorflow as tf
import h5py
import numpy as np

import scipy
from keras import backend as K
from keras.layers import Activation, LSTM, GRU, Dense, Conv1D,\
        MaxPooling1D, GlobalAveragePooling1D, Dropout, Flatten,\
        SimpleRNN, PReLU, BatchNormalization, Conv2D,\
        Conv2DTranspose, MaxPooling2D, GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
import keras.regularizers as regularizers
from keras.optimizers import Adam, Nadam, SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, TerminateOnNaN,\
        ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from keras.utils import plot_model

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

class FinalProjectEEG:
    ''' '''

    def __init__(self):
        ''' '''
        #Reproducible behavior in python 3.2.3+
        os.environ['PYTHONHASHSEED'] = '0'

        #suppress AVX/FMA warning
        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.seed = 0
        #fixed numpy random numbers
        np.random.seed(self.seed)

        #starting core Python generated random numbers
        rn.seed(self.seed)

        self.process_count = 0
        self.session_conf = tf.ConfigProto(intra_op_parallelism_threads=self.process_count,\
                inter_op_parallelism_threads=self.process_count)

        #Well-defined tensorflow backend
        tf.set_random_seed(self.seed)

        self.session = tf.Session(graph=tf.get_default_graph(), config=self.session_conf)
        K.set_session(self.session)

        #class variables
        self.subject_dataset, self.subject_labelset = [], []
        self.data, self.labels = None, None

        #constants
        self.subject_count = 9
        self.num_trials = 50
        self.features = 22
        self.label_count = 4
        self.model_code = ""

        self.TIMESTAMP = self.get_timestamp()

        self.output_dir = ".output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_timestamp(self):
        ''' return formatted timestamp '''
        return datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%dT%H%M%S')

    def log(self, x):
        ''' fancy print statement '''
        d = self.get_timestamp()
        print("%s: %s" %(d, x))

    def read_file(self, fileName):
        ''' Read the input file '''

        self.log("Reading %s" % fileName)
        f = h5py.File(fileName, 'r')
        X = np.copy(f['image'])
        N = X.shape[0]

        X = X[:, :self.features, :]
        X = np.swapaxes(X,1,2)

        y = np.copy(f['type'])
        y = y[0, 0:N:1]
        y = np.asarray(y, dtype=np.int32)

        valid = []
        #np.nan_to_num(X, 0)
        for i in range(N):
            if not np.isnan(X[i]).any():
                valid.append(i)

        X = X[valid]
        y = y[valid]

        return X, y

    def read_files(self):
        ''' Read all files and store them in member variables '''

        for i in range(1, self.subject_count + 1):
            X, y = self.read_file("A0%sT_slice.mat" % i)
            self.subject_dataset.append(X)
            self.subject_labelset.append(y)

        self.data = np.concatenate(self.subject_dataset)
        self.labels = np.concatenate(self.subject_labelset)

    def split_train_test(self, _X, _y):
        ''' Create test sets from all subjects '''
        X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=self.num_trials)
        return X_train, X_test, y_train, y_test

    def normalize_data(self, _data):
        ''' Zero mean unit varience for the data '''

        self.log("Start winsorizing data")
        scipy.stats.mstats.winsorize(_data, limits=0.05, axis=1, inplace=True)
        self.log("Done winsorizing data")

        #Fourier transform
        _data = np.real(np.fft.fft(_data))

        #Normalizes across everything
        #data = self.session.run(tf.image.per_image_standardization(_data))

        #Normalizes across every column independently
        data = (_data - _data.mean(axis=1, keepdims=True)) / (_data.std(axis=1, keepdims=True))

        return data

    def normalize_labels(self, _labels):
        ''' Min-subtraction for the labels '''
        min_val = min(_labels)
        labels = self.session.run(tf.subtract(_labels, min_val))
        return labels

    def preprocess_data(self, data, labels):
        ''' split, normalize '''

        X_train, X_test, y_train, y_test = self.split_train_test(data, labels)

        y_train = self.normalize_labels(y_train)
        y_test = self.normalize_labels(y_test)

        #onehotencoding
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        X_train = self.normalize_data(X_train)
        X_test = self.normalize_data(X_test)

        return X_train, X_test, y_train, y_test

    def run_model(self, X_train, y_train):
        ''' Create the model, train, etc'''

        dim1,dim2 = X_train.shape[1], X_train.shape[2]

        self.log("Creating the model...")
        model = Sequential()

        """ <MODELCODE> """
        epoch_count = 1
        batch_size = 16
        model.add(Conv1D(128, 3, activation='relu', input_shape=(dim1, dim2)))
        model.add(LSTM(64, activation='tanh', return_sequences=True))
        model.add(MaxPooling1D(3))
        model.add(GaussianNoise(stddev=0.5))
        model.add(Flatten())
        model.add(Dropout(0.5))

        model.add(Dense(self.label_count, activation='softmax'))

        opt = Adam()
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        """ </MODELCODE> """

        model.summary()

        #Set callbacks...
        _patience = min(30, max(epoch_count//5, 20))
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                       patience=_patience, verbose=1, mode='auto')
        tn = TerminateOnNaN()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_lr=1e-7, verbose=1)
        csv_logger = CSVLogger(os.path.join(self.output_dir, '%s_training.log' % self.TIMESTAMP))
        checkpoint_path = os.path.join(self.output_dir, "weights.best.hdf5")
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callback_fns = [early_stopping, tn, csv_logger, checkpoint, reduce_lr]

        self.log("Training...")
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch_count, shuffle=False, \
                validation_split=0.1, verbose=1, callbacks=callback_fns)

        return model, history

    def plot(self, model, history, X_train, y_train):
        ''' Plot the graph '''

        output_dir = self.output_dir

        plot_model(model, to_file=os.path.join(output_dir, "%s_model.png" % self.TIMESTAMP), show_shapes=True)
        with open(os.path.join(output_dir, "%s_model.json" % self.TIMESTAMP), 'w') as fp:
            json.dump(model.to_json(), fp)

        history_dict = history.history
        d = self.TIMESTAMP

        def _graph(plt, d, _x, _y, quantity):
            self.log("Plotting the %s graph..." % quantity)
            epochs = range(1, len(_x) + 1)
            plt.figure()
            plt.plot(epochs, _x, 'b', label='Training %s' % quantity)
            plt.plot(epochs, _y, 'r', label='Validation %s' % quantity)

            plt.title('Training and validation %s' % quantity)
            plt.xlabel('epoch')
            plt.ylabel(quantity)
            plt.legend()
            #plt.show()

            file_name = os.path.join(output_dir, "%s_%s_graph.png" %(d, quantity))
            plt.savefig(file_name, bbox_inches='tight')
            self.log("Graph saved as %s" % file_name)

        #Plot loss
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        _graph(plt, d, loss_values, val_loss_values, "loss")

        #Plot accuracy
        acc_values = history_dict['acc']
        val_acc_values = history_dict['val_acc']
        _graph(plt, d, acc_values, val_acc_values, "accuracy")

    def evaluate(self, model, X_test, y_test):
        ''' Evaluate the model '''
        self.log("Evaluating...")
        loss, acc = model.evaluate(X_test, y_test)
        self.log("Loss: %s, Accuracy: %s" %(loss, acc))
        return loss, acc

    def cleanup(self):
        ''' Cleanup '''
        K.clear_session()

    def run(self):
        ''' RUN! '''

        start = datetime.datetime.now()
        self.log("Starting...")
        self.read_files()

        #Try on a per-subject basis if needed
        #subject = 0
        #s_data, s_labels = self.subject_dataset[subject], self.subject_labelset[subject]

        s_data, s_labels = self.data, self.labels

        X_train, X_test, y_train, y_test = self.preprocess_data(s_data, s_labels)
        model, history = self.run_model(X_train, y_train)

        training_loss, training_acc = self.evaluate(model, X_test, y_test)

        self.plot(model, history, X_train, y_train)

        #Non-essentials follow
        print("")
        end = datetime.datetime.now()
        diff = end-start
        self.log("Time taken: %s" %(diff))

        self.log("Done...")

        self.cleanup()

if __name__ == "__main__":
    ''' '''
    eegdata = FinalProjectEEG()
    eegdata.run()
