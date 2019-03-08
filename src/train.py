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
from utils import normalize_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_one_hot(targets, nb_classes):
    """Convert class array to one hot vector."""
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def train(dataset, model_name, timestep=240):
    """Train an LSTM model."""
    for i in range(len(dataset[0])):
        model_period = f"{model_name}_period{i}.h5"
        x_train = normalize_data(dataset[0][i][0])
        y_train = get_one_hot(dataset[0][i][1].values, 2) * 1.0
        x_test = normalize_data(dataset[1][i][0])
        y_test = get_one_hot(dataset[1][5][1].values, 2) * 1.0

        train_gen = TimeseriesGenerator(x_train, y_train,
                                        length=timestep, sampling_rate=1,
                                        batch_size=64)
        test_gen = TimeseriesGenerator(x_test, y_test,
                                       length=timestep, sampling_rate=1,
                                       batch_size=64)
        print(f"Period {i}")
        print(f"x train shape: {x_train.shape}")
        print(f"y train shape: {y_train.shape}")
        print(f"x test shape: {x_test.shape}")
        print(f"y test shape: {y_test.shape}")

        # expected input data shape: (batch_size, timesteps, data_dim)
        regressor = Sequential()
        regressor.add(LSTM(units=25, return_sequences=True,
                           input_shape=(timestep, 31), dropout=0.1))
        regressor.add(LSTM(25, dropout=0.1))
        regressor.add(Dense(62, activation='relu'))
        regressor.add(Reshape((31, 2)))
        regressor.add(Lambda(lambda x: softmax(x, axis=-1)))
        regressor.compile(loss='binary_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'])

        regressor.fit_generator(train_gen, steps_per_epoch=len(train_gen),
                                epochs=10, validation_data=test_gen,
                                callbacks=[
                                    EarlyStopping(monitor='val_loss',
                                                  mode='min', patience=10),
                                    ModelCheckpoint(filepath=model_period,
                                                    monitor='val_acc',
                                                    save_best_only=True)])

        # regressor.save(model_period)


def main():
    """Run main program."""
    parser = argparse.ArgumentParser(
        description="Parse arguments for models.")
    parser.add_argument("--dataset", help="Dataset directory.",
                        default="../data/dowjones_calculated/periods.txt")
    parser.add_argument('--outdir', help='Model directory.',
                        default='../model/LSTM/my_model2')
    args = parser.parse_args()

    with open(args.dataset, "rb") as file:   # Unpickling
        dataset = pickle.load(file)
    train(dataset, args.outdir)

    print("Done.")
    return 0


if __name__ == "__main__":
    main()
