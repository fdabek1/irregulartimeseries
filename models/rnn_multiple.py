from models.chunk_model import ChunkModel
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# This class uses each timestep's output as a prediction.
class RNNMultiple(ChunkModel):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.scaler = MinMaxScaler()

        self.train_add = -1
        self.test_add = -1

    # Override the way that ChunkModel chunks everything together
    def transform(self):
        num_features = len(self.analysis.features)

        self.train_add = self.num_days - (len(self.analysis.x_train) % self.num_days)
        self.x_train = np.concatenate((self.analysis.x_train, (np.ones((self.train_add, num_features)) * 0)))
        self.y_train = np.concatenate((self.analysis.y_train, (np.ones((self.train_add,)) * 0)))

        self.test_add = self.num_days - (len(self.analysis.x_test) % self.num_days)
        self.x_test = np.concatenate((self.analysis.x_test, (np.ones((self.test_add, num_features)) * 0)))
        self.y_test = np.concatenate((self.analysis.y_test, (np.ones((self.test_add,)) * 0)))

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
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), verbose=2,
                       **self.fit_config)

    def predict(self):
        def run_clean(x, num_extra):
            # Run the model
            output = self.model.predict(x).flatten()

            # Exclude extra predictions
            output = output[:-1 * num_extra]

            # Scale back to original values
            output = self.scaler.inverse_transform(output.reshape(-1, 1)).flatten()

            return output

        return run_clean(self.x_train, self.train_add), run_clean(self.x_test, self.test_add)
