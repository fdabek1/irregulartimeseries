from models.model import Model
from data.transformations import normalize_y
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# This class uses each timestep's output as a prediction.
class RNNMultiple(Model):
    def __init__(self, analysis, model, num_days=5, mask_value=None):
        super().__init__(analysis)
        self.model = model
        self.scaler = MinMaxScaler()

        self.num_days = num_days
        self.mask_value = mask_value

    def transform(self):
        num_features = len(self.analysis.features)

        train_add = self.num_days - (len(self.analysis.x_train) % self.num_days)
        self.x_train = np.concatenate((self.analysis.x_train, (np.ones((train_add, num_features)) * self.mask_value)))
        self.y_train = np.concatenate((self.analysis.y_train, (np.ones((train_add,)) * self.mask_value)))

        test_add = self.num_days - (len(self.analysis.x_test) % self.num_days)
        self.x_test = np.concatenate((self.analysis.x_test, (np.ones((test_add, num_features)) * self.mask_value)))
        self.y_test = np.concatenate((self.analysis.y_test, (np.ones((test_add,)) * self.mask_value)))

        self.x_train = self.x_train.reshape((len(self.x_train) // self.num_days, self.num_days, num_features))
        self.y_train = self.y_train.reshape((len(self.y_train) // self.num_days, self.num_days, 1))

        self.x_test = self.x_test.reshape((len(self.x_test) // self.num_days, self.num_days, num_features))
        self.y_test = self.y_test.reshape((len(self.y_test) // self.num_days, self.num_days, 1))

        # Normalize the y variable between 0 and 1
        self.scaler.fit(np.concatenate((self.y_train, self.y_test)).reshape((-1, 1)))
        self.y_train = self.scaler.transform(self.y_train.reshape(-1, 1)).reshape((-1, self.num_days, 1))
        self.y_test = self.scaler.transform(self.y_test.reshape(-1, 1)).reshape((-1, self.num_days, 1))

    def build_model(self):
        pass

    def train(self):
        self.model.fit(self.x_train, self.y_train, epochs=10, validation_data=(self.x_test, self.y_test),
                       batch_size=10,
                       verbose=2)

    def predict(self):
        def run_clean(x):
            # Run the model
            output = self.model.predict(x).flatten()

            # Find where x is not set to the mask value
            indices = np.where(x.flatten() != self.mask_value)

            # Get the output for those locations
            output = output[indices]

            # Scale back to original values
            output = self.scaler.inverse_transform(output.reshape(-1, 1)).flatten()

            return output

        return run_clean(self.x_train), run_clean(self.x_test)
