#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train an test a random forest model."""

import argparse
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from utils import generate_time_series_sample, normalize_data


def train(dataset, model_name, timestep=20):
    """Train an LSTM model."""
    positions = []
    for i in range(len(dataset[0])):
        # model_period = f"{model_name}_period{i}.h5"

        x_train, y_train = generate_time_series_sample(
            normalize_data(dataset[0][i][0]),
            dataset[0][i][1].values, 20)

        x_test, y_test = generate_time_series_sample(
            normalize_data(dataset[1][i][0]),
            dataset[1][i][1].values, 20)

        x_train = x_train.transpose((0, 2, 1))
        x_train = np.reshape(
            x_train, (x_train.shape[0] * x_train.shape[1], timestep))
        y_train = np.reshape(y_train, (y_train.shape[0] * y_train.shape[1]))

        x_test = x_test.transpose((0, 2, 1))
        x_test = np.reshape(
            x_test, (x_test.shape[0] * x_test.shape[1], timestep))
        y_test = np.reshape(y_test, (y_test.shape[0] * y_test.shape[1]))
        print(f"x train shape: {x_train.shape}")
        print(f"y train shape: {y_train.shape}")
        print(f"x test shape: {x_test.shape}")
        print(f"y test shape: {y_test.shape}")

        clf = RandomForestClassifier(n_jobs=2, random_state=0, max_depth=5)
        clf.fit(x_train, y_train)
        predict = clf.predict(x_test)
        predict = predict.reshape(predict.shape[0] // 31, 31)[-250:]
        position = dataset[1][i][1].values[-250:, :]
        result = sum(sum(predict == position)) / predict.size

        predict1 = clf.predict(x_test)
        predict1 = predict1.reshape(predict1.shape[0] // 31, 31)[-300:-250]
        position1 = dataset[1][i][1].values[-300:-250, :]
        result1 = sum(sum(predict1 == position1)) / predict1.size

        positions.append(predict)
        print(result)
        print(result1)
    all_positions = np.concatenate(positions, axis=0)
    print(all_positions.shape)


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
