from models.chunk_model import ChunkModel
from sklearn.preprocessing import MinMaxScaler
from .nn_chart_helper import draw_nn_log
import numpy as np


# This class only takes the final output as a prediction
class RNNSingle(ChunkModel):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.scaler = MinMaxScaler()

    def transform(self):
        super().transform()

        # Put the chunks into time steps
        self.x_train = self.x_train.reshape((-1, self.num_days, len(self.analysis.features)))
        self.x_test = self.x_test.reshape((-1, self.num_days, len(self.analysis.features)))

        # Scale the y values
        self.scaler.fit(np.concatenate((self.y_train, self.y_test)).reshape((-1, 1)))
        self.y_train = self.scaler.transform(self.y_train.reshape(-1, 1)).flatten()
        self.y_test = self.scaler.transform(self.y_test.reshape(-1, 1)).flatten()

    def build_model(self):
        pass

    def train(self):
        history = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), verbose=0,
                                 **self.fit_config)
        draw_nn_log(history)

    def predict(self):
        return self.predict_transform(self.scaler.inverse_transform(self.model.predict(self.x_train)).flatten(),
                                      self.scaler.inverse_transform(self.model.predict(self.x_test)).flatten())
