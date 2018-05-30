from analysis import Analysis
from models import *

import numpy as np

np.random.seed(7)

analysis = Analysis('passengers', percent_train=0.67, normalize=True)
analysis.load_data()

# Move this into a function to allow for quickly switching between predictors and features
# Or just leave it since will do our own feature selection?
predictor = 'num_passengers'
features = [predictor]
analysis.set_feature_predictor_columns(features, predictor)

analysis.train_test_split()

NUM_DAYS = 3
NUM_FEATURES = len(analysis.features)

from keras.models import Sequential
from keras import layers

nn = Sequential()
nn.add(layers.LSTM(50, activation='tanh', input_shape=(len(analysis.x_train), NUM_FEATURES), return_sequences=True))

nn.add(layers.TimeDistributed(layers.Dense(10, activation='relu')))
nn.add(layers.TimeDistributed(layers.Dense(1, activation='tanh')))
# nn.add(layers.TimeDistributed(layers.Dense(1, activation='linear')))
nn.compile(loss='mse', optimizer='adam')

model = RNNAll(nn, analysis=analysis, fit_config={'epochs': 30})
analysis.run_model(model, name='RNNAll')
