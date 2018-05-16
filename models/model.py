class Model:
    def __init__(self, analysis):
        self.analysis = analysis
        self.model = None

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def transform(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def train(self):
        if self.x_train is None or self.y_train is None:
            self.model.fit(self.analysis.x_train, self.analysis.y_train)
        else:
            self.model.fit(self.x_train, self.y_train)

    def predict(self):
        if self.x_train is None or self.x_test is None:
            return self.model.predict(self.analysis.x_train), self.model.predict(self.analysis.x_test)
        else:
            return self.model.predict(self.x_train), self.model.predict(self.x_test)
