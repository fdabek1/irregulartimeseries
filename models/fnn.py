from models.chunk_model import ChunkModel
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class FNN(ChunkModel):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.scaler = MinMaxScaler()

    def transform(self):
        super().transform()

        # Scale the y values
        self.scaler.fit(np.concatenate((self.y_train, self.y_test)).reshape((-1, 1)))
        self.y_train = self.scaler.transform(self.y_train.reshape(-1, 1)).flatten()
        self.y_test = self.scaler.transform(self.y_test.reshape(-1, 1)).flatten()

    def build_model(self):
        pass

    def train(self):
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), verbose=2,
                       **self.fit_config)

    def predict(self):
        return self.predict_transform(self.scaler.inverse_transform(self.model.predict(self.x_train)).flatten(),
                                      self.scaler.inverse_transform(self.model.predict(self.x_test)).flatten())
