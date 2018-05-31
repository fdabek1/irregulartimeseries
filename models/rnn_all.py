from models.model import Model
from sklearn.preprocessing import MinMaxScaler
from .nn_chart_helper import draw_nn_log
import numpy as np


# This class has only one sample of the entire dataset
class RNNAll(Model):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.scaler = MinMaxScaler()

        self.test_add = -1

    def transform(self):
        self.x_train = self.analysis.x_train
        self.y_train = self.analysis.y_train
        self.x_test = self.analysis.x_test
        self.y_test = self.analysis.y_test

        # Add dummy values to the test data so that it
        num_features = len(self.analysis.features)
        self.test_add = len(self.x_train) - len(self.x_test)
        self.x_test = np.concatenate((self.x_test, [[0] * num_features] * self.test_add))
        self.y_test = np.concatenate((self.y_test, [0] * self.test_add))

        # Normalize the y variable between 0 and 1
        self.scaler.fit(np.concatenate((self.y_train, self.y_test)).reshape((-1, 1)))
        self.y_train = self.scaler.transform(self.y_train.reshape(-1, 1))
        self.y_test = self.scaler.transform(self.y_test.reshape(-1, 1))

        # Change into each being a single sample
        self.x_train = np.asarray([self.x_train])
        self.y_train = np.asarray([self.y_train])
        self.x_test = np.asarray([self.x_test])
        self.y_test = np.asarray([self.y_test])

    def build_model(self):
        pass

    def train(self):
        history = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), verbose=0,
                                 **self.fit_config)
        draw_nn_log(history)

    def predict(self):
        # Run the model
        test_output = self.model.predict(self.x_test).flatten()

        # Exclude extra predictions
        test_output = test_output[:-1 * self.test_add]

        # Scale back to original values
        test_output = self.scaler.inverse_transform(test_output.reshape(-1, 1)).flatten()

        return self.scaler.inverse_transform(self.model.predict(self.x_train)[0]).flatten(), test_output
