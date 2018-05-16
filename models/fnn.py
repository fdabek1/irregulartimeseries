from models.chunk_model import ChunkModel
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class FNN(ChunkModel):
    def __init__(self, analysis, model, num_days=5):
        super().__init__(analysis, num_days)
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
        self.model.fit(self.x_train, self.y_train, epochs=10, validation_data=(self.x_test, self.y_test), batch_size=10,
                       verbose=2)

    def predict(self):
        return self.predict_transform(self.scaler.inverse_transform(self.model.predict(self.x_train)).flatten(),
                                      self.scaler.inverse_transform(self.model.predict(self.x_test)).flatten())
