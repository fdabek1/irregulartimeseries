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
from keras import optimizers
from keras import layers
np.random.seed(7)

nn = Sequential()
nn.add(layers.LSTM(4, input_shape=(NUM_DAYS, NUM_FEATURES)))

nn.add(layers.Dense(1))
nn.compile(loss='mean_squared_error', optimizer='adam')

model = RNNSingle(analysis=analysis, model=nn, num_days=NUM_DAYS)
analysis.run_model(model, name='LSTMSingle')
