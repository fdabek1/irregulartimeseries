from analysis import Analysis
from models import *

analysis = Analysis('weather')
analysis.load_data()

# Move this into a function to allow for quickly switching between predictors and features
# Or just leave it since will do our own feature selection?
predictor = 'actual_mean_temp'
features = list(analysis.data.columns)
features.remove('date')
features.remove(predictor)
features = [predictor]
analysis.set_feature_predictor_columns(features, predictor)

analysis.train_test_split()

NUM_DAYS = 5
NUM_FEATURES = len(analysis.features)

# model = LinearRegression(analysis, num_days=NUM_DAYS)
# analysis.run_model(model)
#
# exit()

from keras.models import Sequential
from keras import optimizers
from keras import layers

mask_value = -9999

nn = Sequential()
nn.add(layers.Masking(mask_value=mask_value, input_shape=(len(analysis.x_train), NUM_FEATURES)))
nn.add(layers.LSTM(50, activation='tanh', input_shape=(len(analysis.x_train), NUM_FEATURES), return_sequences=True))

nn.add(layers.TimeDistributed(layers.Dense(10, activation='relu')))
nn.add(layers.TimeDistributed(layers.Dense(1, activation='tanh')))
# nn.add(layers.TimeDistributed(layers.Dense(1, activation='linear')))
nn.compile(loss='mae', optimizer='rmsprop')

model = RNNAll(analysis, nn, mask_value=mask_value)
analysis.run_model(model)

exit()
