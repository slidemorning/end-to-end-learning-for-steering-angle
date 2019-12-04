import numpy as np
import random
import h5py
import pandas as pd
import matplotlib.pyplot as plt

X_PATH = './data/image/img_preprocessed_v1.h5'
Y_PATH = './data/label/label.h5'
X_KEY = 'X'
Y_KEY = 'steering_angle'
IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL = 200, 66, 1

# internal
def __down_sampling__(data, fro, to):
  ret = []
  inc = fro//to
  for i in range(inc*10000, inc*20000, inc):
    ret.append(data[i])
  return np.array(ret)

# load dataset
def load_data(x_path=X_PATH, y_path=Y_PATH):
  h5_x = h5py.File(x_path, 'r')
  h5_y = h5py.File(y_path, 'r')
  X = h5_x[X_KEY] # (10000, 66, 200, 1)
  y = h5_y[Y_KEY] # (10000,)
  y = __down_sampling__(y, 100, 20)
  return X, y

# shuffle dataset
def shuffle_data(X, y):
  pair = [[a, b] for a, b in zip(X, y)]
  random.shuffle(pair)
  X_sh = [p[0] for p in pair]
  y_sh = [p[1] for p in pair]
  X_sh = np.array(X_sh)
  y_sh = np.array(y_sh)
  return X_sh, y_sh

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(8, 12))

  plt.subplot(2, 1, 1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label='Val Error')
  plt.ylim([0, 10])
  plt.legend()

  plt.subplot(2, 1, 2)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label='Val Error')
  plt.ylim([0, 100])
  plt.legend()
  plt.show()

if __name__ == '__main__':
    pass
