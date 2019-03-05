#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train a LSTM model."""

import argparse
import os
import pickle

import numpy as np

from keras.activations import softmax
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dense, Lambda, Reshape
from keras.models import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_one_hot(targets, nb_classes):
    """Convert class array to one hot vector."""
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def train(dataset, model_name):
    """Train an LSTM model."""
    for i in range(len(dataset[0])):
        model_period = f"{model_name}_period{i}.h5"
        X_train = dataset[0][i][0].values
        y_train = dataset[0][i][1].values
        # X_test = dataset[1][i][0].values
        # y_test = dataset[1][i][1].values
        y_train = get_one_hot(y_train, 2)
        # y_test = get_one_hot(y_test, 2)

        train_gen = TimeseriesGenerator(X_train, y_train,
                                        length=240, sampling_rate=1,
                                        batch_size=510)
        # test_gen = TimeseriesGenerator(X_test, y_test,
        #                                length=240, sampling_rate=1,
        #                                batch_size=250)

        X_train = train_gen[0][0]
        y_train = train_gen[0][1]
        # X_test = test_gen[0][0]
        # y_test = test_gen[0][1]

        print(f"x train shape: {X_train.shape}")
        print(f"y train shape: {y_train.shape}")
        # print(f"x test shape: {X_test.shape}")
        # print(f"y test shape: {y_test.shape}")

        # expected input data shape: (batch_size, timesteps, data_dim)
        regressor = Sequential()
        regressor.add(LSTM(units=10, input_shape=(
            X_train.shape[1], X_train.shape[2]), dropout=0.1))
        regressor.add(Dense(62, activation='relu'))
        regressor.add(Reshape((31, 2)))
        regressor.add(Lambda(lambda x: softmax(x, axis=-1)))
        regressor.compile(loss='mean_squared_error',
                          optimizer='rmsprop',
                          metrics=['accuracy'])

        regressor.fit(X_train, y_train, epochs=100, batch_size=10,
                      validation_split=0.1,
                      callbacks=[EarlyStopping(monitor='val_acc', patience=50),
                                 ModelCheckpoint(filepath=model_period,
                                                 monitor='val_acc',
                                                 save_best_only=True)])

        regressor.save(model_period)


def main():
    """Run main program."""
    parser = argparse.ArgumentParser(
        description="Parse arguments for models.")
    parser.add_argument("--dataset", help="Dataset directory.",
                        default="../data/dowjones_calculated/periods.txt")
    parser.add_argument('--outdir', help='Model directory.',
                        default='../model/LSTM/my_model1')
    args = parser.parse_args()

    with open(args.dataset, "rb") as file:   # Unpickling
        dataset = pickle.load(file)
    train(dataset, args.outdir)

    print("Done.")
    return 0


if __name__ == "__main__":
    main()
