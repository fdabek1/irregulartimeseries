from models.model import Model
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# This class has only one sample of the entire dataset
class RNNAll(Model):
    def __init__(self, analysis, model, num_days=5, mask_value=None):
        super().__init__(analysis)
        self.model = model
        self.scaler = MinMaxScaler()

        self.num_days = num_days
        self.mask_value = mask_value

    def transform(self):
        self.x_train = self.analysis.x_train
        self.y_train = self.analysis.y_train
        self.x_test = self.analysis.x_test
        self.y_test = self.analysis.y_test

        # Add dummy values to the test data so that it
        num_features = len(self.analysis.features)
        diff = len(self.x_train) - len(self.x_test)
        self.x_test = np.concatenate((self.x_test, [[self.mask_value] * num_features] * diff))
        self.y_test = np.concatenate((self.y_test, [self.mask_value] * diff))

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
        self.model.fit(self.x_train, self.y_train, epochs=10, validation_data=(self.x_test, self.y_test),
                       batch_size=10,
                       verbose=2)

    def predict(self):
        # Run the model
        test_output = self.model.predict(self.x_test).flatten()

        # Find where x is not set to the mask value
        indices = np.where(self.x_test.flatten() != self.mask_value)

        # Get the output for those locations
        test_output = test_output[indices]

        # Scale back to original values
        test_output = self.scaler.inverse_transform(test_output.reshape(-1, 1)).flatten()

        return self.scaler.inverse_transform(self.model.predict(self.x_train)[0]).flatten(), test_output
