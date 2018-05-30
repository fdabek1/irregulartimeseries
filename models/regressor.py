from models.model import Model


# A catch-all for all types of regressor's in sci-kit learn
class Regressor(Model):
    def __init__(self, regressor, params=None, **kwargs):
        super().__init__(**kwargs)

        self.regressor = regressor
        self.params = params

    def transform(self):
        pass

    def build_model(self):
        self.model = self.regressor()
