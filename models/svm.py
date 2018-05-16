from models.chunk_model import ChunkModel
from sklearn.svm import SVR


class SVM(ChunkModel):
    def build_model(self):
        self.model = SVR()
