import matplotlib.pyplot as plt
import numpy as np


def draw_nn_log(history):
    num_epochs = len(history.history['loss'])
    plt.plot(np.arange(num_epochs), history.history['loss'], label='train')
    plt.plot(np.arange(num_epochs), history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
