from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os

plt.rcParams['figure.figsize'] = [20, 10]  # https://stackoverflow.com/a/36368418/556935
# print(plt.style.available)
plt.style.use('fivethirtyeight')
pd.options.mode.chained_assignment = None  # default='warn'


class Analysis:
    def __init__(self, data_type, normalize=False, missing_type=None, missing_percent=0.2, missing_seed=3,
                 logger=print):
        self.data_type = data_type
        self.data = None
        self.logger = logger
        self.normalize = normalize

        self.missing_type = missing_type
        self.missing_percent = missing_percent
        self.missing_seed = missing_seed

        self.num_train = -1

        self.features = None
        self.predictor = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load_data(self):
        folder = os.getcwd()

        if self.data_type == 'weather':
            self.data = pd.read_csv(folder + '/data/weather/KPHL.csv', parse_dates=['date'])
        else:
            raise Exception('Invalid data type.')

        if self.missing_type is not None:
            random.seed(self.missing_seed)
            if self.missing_type == 'remove':
                self.remove_time_points()

    def remove_time_points(self):
        data_len = len(self.data)
        num_remove = int(round(data_len * self.missing_percent))
        indices_remove = [random.randint(0, data_len - 1) for _ in range(num_remove)]

        self.data = self.data.drop(indices_remove)

    def set_feature_predictor_columns(self, features, predictor):
        self.features = features
        self.predictor = predictor

    def train_test_split(self):
        # Split into x and y
        x = self.data[self.features].iloc[:-1]
        y = self.data[self.predictor][1:]

        if self.normalize:
            x = self.normalize_features(x)

        data_len = len(self.data)
        num_train = int(round(data_len * 0.8))
        self.num_train = num_train

        self.logger('Number of train data points: ' + str(num_train))
        self.logger('Number of test data points: ' + str(data_len - num_train))

        # Split into training and test data
        self.x_train = x[:num_train].as_matrix()
        self.y_train = y[:num_train].as_matrix()
        self.x_test = x[num_train:].as_matrix()
        self.y_test = y[num_train:].as_matrix()

    @staticmethod
    def normalize_features(x):
        for col in x.columns:
            scaler = MinMaxScaler()
            x[col] = scaler.fit_transform(x[[col]])

        return x

    def run_model(self, model):
        model.transform()
        model.build_model()
        model.train()

        predicted_train, predicted_test = model.predict()
        self.compute_error(predicted_train, predicted_test)
        self.visualize_data(predicted_train, predicted_test)

    def compute_error(self, predicted_train, predicted_test):
        assert len(predicted_train) == len(self.y_train)
        assert len(predicted_test) == len(self.y_test)

        # Find NaN indices
        indices_train = np.argwhere(~np.isnan(predicted_train)).flatten()
        indices_test = np.argwhere(~np.isnan(predicted_test)).flatten()

        # Find errors ignoring NaN's
        error_train = mean_absolute_error(self.y_train[indices_train], predicted_train[indices_train])
        error_test = mean_absolute_error(self.y_test[indices_test], predicted_test[indices_test])

        self.logger('Train Error: ' + str(error_train) + '  Num NaN: ' + str(len(self.y_train) - len(indices_train)))
        self.logger('Test Error: ' + str(error_test) + '    Num NaN: ' + str(len(self.y_test) - len(indices_test)))

    def visualize_data(self, predicted_train=None, predicted_test=None, show_raw=True, show_diff=False):
        if show_raw:
            # plt.title('RNN - Weather')
            plt.plot(np.arange(len(self.y_train)), self.y_train, label='train')
            plt.plot(len(self.y_train) + np.arange(len(self.y_test)), self.y_test, label='test')
            if predicted_train is not None:
                predicted = np.concatenate((predicted_train, predicted_test))
                plt.plot(np.arange(len(predicted)), predicted, label='predict')
            plt.legend()
            plt.show()

        if show_diff:
            # plt.title('Differences')
            diff_train = predicted_train - self.y_train
            diff_test = predicted_test - self.y_test

            # plt.scatter(np.arange(len(diff_train)), diff_train, s=1, alpha=0.5)
            fig, ax = plt.subplots()
            ax.fill(np.arange(len(diff_train)), diff_train, '#000000', len(diff_train) + np.arange(len(diff_test)),
                    diff_test, 'r')

            # ax.fill(x, y, zorder=10)
            ax.grid(True, zorder=5)
            plt.show()

            exit()
            # plt.gcf().clear()
            # # plt.title('Differences')
            # # diff_train = predicted_train - self.y_train[:len(predicted_train)]
            # # diff_test = predicted_test - self.y_test[:len(predicted_test)]
            #
            # x = np.linspace(0, 1, 500)
            # y = np.sin(4 * np.pi * x) * np.exp(-5 * x)
            #
            # fig, ax = plt.subplots()
            #
            # ax.fill(x, y, zorder=10)
            # ax.grid(True, zorder=5)
            #
            # # print(diff_train)
            #
            # # ax.fill(np.arange(len(diff_train)), diff_train, zorder=10)
            #
            # # plt.plot(np.arange(len(self.y_train)), self.y_train, label='train')
            # # plt.plot(len(self.y_train) + np.arange(len(self.y_test)), self.y_test, label='test')
            # # if predicted_train is not None:
            # #     plt.plot(len(self.y_train) + np.arange(len(self.y_test)), predicted_train, label='predict')
            # # plt.legend()
            # plt.show()
            # exit()
