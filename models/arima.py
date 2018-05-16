from models.model import Model
from pyramid.arima import auto_arima


class ARIMA(Model):
    def __init__(self, analysis, use_features=True):
        super().__init__(analysis)

        self.use_features = use_features

    def transform(self):
        pass

    def build_model(self):
        if self.use_features:
            self.model = auto_arima(self.analysis.y_train, self.analysis.x_train, start_p=1, start_q=1, max_p=3,
                                    max_q=3, m=12,
                                    start_P=0, seasonal=True, d=1, D=1, trace=True,
                                    error_action='ignore',  # don't want to know if an order does not work
                                    suppress_warnings=True,  # don't want convergence warnings
                                    stepwise=True)  # set to stepwise
        else:
            self.model = auto_arima(self.analysis.y_train, start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                                    start_P=0, seasonal=True, d=1, D=1, trace=True,
                                    error_action='ignore',  # don't want to know if an order does not work
                                    suppress_warnings=True,  # don't want convergence warnings
                                    stepwise=True)  # set to stepwise

        # print(self.model.summary())

    def train(self):
        pass

    def predict(self):
        if self.use_features:
            return self.model.predict_in_sample(exogenous=self.analysis.x_train), \
                   self.model.predict(exogenous=self.analysis.x_test, n_periods=len(self.analysis.y_test))
        else:
            return self.model.predict_in_sample(), self.model.predict(n_periods=len(self.analysis.y_test))
