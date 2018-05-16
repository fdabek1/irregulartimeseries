from models.chunk_model import ChunkModel
from sklearn import linear_model


class LinearRegression(ChunkModel):
    def build_model(self):
        self.model = linear_model.LinearRegression()
