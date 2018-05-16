from models.model import Model
import numpy as np


class ChunkModel(Model):
    def __init__(self, analysis, num_days):
        super().__init__(analysis)

        self.num_days = num_days

    def transform(self):
        num_features = len(self.analysis.features)
        self.x_train = np.empty((0, num_features * self.num_days))
        self.y_train = np.empty((0,))
        self.x_test = np.empty((0, num_features * self.num_days))
        self.y_test = np.empty((0,))

        data_len = len(self.analysis.data)
        for t in range(self.num_days, self.analysis.num_train):
            self.x_train = np.concatenate((self.x_train, [self.analysis.x_train[t - self.num_days:t].flatten()]))
            self.y_train = np.concatenate((self.y_train, [self.analysis.y_train[t]]))

        for t in range(self.num_days, data_len - self.analysis.num_train - 1):
            self.x_test = np.concatenate((self.x_test, [self.analysis.x_test[t - self.num_days:t].flatten()]))
            self.y_test = np.concatenate((self.y_test, [self.analysis.y_test[t]]))

    def build_model(self):
        raise NotImplementedError

    # Add nan's to the predictions made by the ChunkModel
    def predict_transform(self, train, test):
        return np.concatenate(([np.nan] * self.num_days, train)), np.concatenate(([np.nan] * self.num_days, test))

    def predict(self):
        return self.predict_transform(*super().predict())
