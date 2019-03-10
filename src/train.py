#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train a LSTM model."""

import argparse
import os
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dense, Reshape
from keras.models import Sequential
from keras.utils import to_categorical

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_one_hot(targets, nb_classes):
    """Convert class array to one hot vector."""
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def train_one_feature(dataset, model_name, timestep=240, feature=31):
    """Train an LSTM model with 1 feature."""
    for i in range(len(dataset[0])):
        model_period = f"{model_name}_1feature_period{i}.h5"
        x_train = dataset[0][i][0].values
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        y_train = to_categorical(dataset[0][i][1].values, 2)

        print(f"Period {i}")
        print(f"x train shape: {x_train.shape}")
        print(f"y train shape: {y_train.shape}")

        x_series = [x_train[i:i + timestep, j]
                    for i in range(x_train.shape[0] - timestep)
                    for j in range(feature)]
        y_series = [y_train[i + timestep, j]
                    for i in range(y_train.shape[0] - timestep)
                    for j in range(feature)]
        x_final = np.array(x_series)
        y_final = np.array(y_series)
        x_final = np.reshape(x_final, (x_final.shape[0], x_final.shape[1], 1))
        print(f"x_final shape: {x_final.shape}")
        print(f"y_final shape: {y_final.shape}")

        # expected input data shape: (batch_size, timesteps, data_dim)
        regressor = Sequential()
        regressor.add(LSTM(units=25, input_shape=(timestep, 1)))
        regressor.add(Dense(2, activation='softmax'))
        regressor.compile(loss='binary_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'])
        regressor.summary()

        regressor.fit(x_final, y_final, batch_size=1000, epochs=1000,
                      validation_split=0.2,
                      callbacks=[EarlyStopping(monitor='val_loss',
                                               mode='min', patience=10),
                                 ModelCheckpoint(filepath=model_period,
                                                 monitor='val_acc',
                                                 save_best_only=True)])


def train(dataset, model_name, timestep=240, feature=31):
    """Train an LSTM model."""
    for i in range(len(dataset[0])):
        model_period = f"{model_name}_period{i}.h5"
        x_train = dataset[0][i][0].values
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        y_train = to_categorical(dataset[0][i][1].values, 2)

        print(f"Period {i}")
        print(f"x train shape: {x_train.shape}")
        print(f"y train shape: {y_train.shape}")

        x_series = [x_train[i:i + timestep, :]
                    for i in range(x_train.shape[0] - timestep)]
        y_series = [y_train[i + timestep]
                    for i in range(y_train.shape[0] - timestep)]
        x_final = np.array(x_series)
        y_final = np.array(y_series)
        print(f"x_final shape: {x_final.shape}")
        print(f"y_final shape: {y_final.shape}")

        # expected input data shape: (batch_size, timesteps, data_dim)
        regressor = Sequential()
        regressor.add(LSTM(units=25, input_shape=(timestep, feature)))
        regressor.add(Dense(feature * 2, activation='relu'))
        regressor.add(Reshape((feature, 2)))
        regressor.add(Dense(2, activation='softmax'))
        regressor.compile(loss='binary_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'])
        regressor.summary()

        regressor.fit(x_final, y_final, batch_size=1000, epochs=1000,
                      validation_split=0.2,
                      callbacks=[EarlyStopping(monitor='val_loss',
                                               mode='min', patience=10),
                                 ModelCheckpoint(filepath=model_period,
                                                 monitor='val_acc',
                                                 save_best_only=True)])


def main():
    """Run main program."""
    index = "dowjones"
    # index = "frankfurt"
    parser = argparse.ArgumentParser(
        description="Parse arguments for models.")
    # parser.add_argument("--dataset", help="Dataset directory.",
    #                     default=f"../data/{index}_calculated/"
    #                     f"absolute_periods750_250_240.txt")
    # parser.add_argument('--outdir', help='Model directory.',
    #                     default=f'../model/LSTM/{index}2_absolute')
    parser.add_argument("--dataset", help="Dataset directory.",
                        default=f"../data/{index}_calculated/"
                        f"periods750_250_240.txt")
    parser.add_argument('--outdir', help='Model directory.',
                        default=f'../model/LSTM/{index}2_')
    args = parser.parse_args()

    with open(args.dataset, "rb") as file:   # Unpickling
        dataset = pickle.load(file)
    train(dataset, args.outdir)
    train_one_feature(dataset, args.outdir)
    print("Done.")
    return 0


if __name__ == "__main__":
    main()
