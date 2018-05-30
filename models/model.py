class Model:
    def __init__(self, analysis=None, fit_config=None):
        self.analysis = analysis
        self.model = None

        # Set the default fit_config if it does not exist
        # This currently is only used for RNN classes
        # Could be abstracted to work for all if needed in the future.
        self.fit_config = {'batch_size': 1} if fit_config is None else fit_config

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
