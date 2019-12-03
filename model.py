from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D

IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL = 200, 66, 1

def get_model():
  model = Sequential()
  model.add(Lambda(lambda x: x/255.,
                   input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL),
                   output_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)))
  model.add(Convolution2D(8, 5, 5, subsample=(4, 4), border_mode='same'))
  model.add(ELU())
  model.add(Convolution2D(16, 5, 5, subsample=(2, 2), border_mode='same'))
  model.add(ELU())
  model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='same'))
  model.add(Flatten())
  model.add(Dropout(.2))
  model.add(ELU())
  model.add(Dense(512))
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(1))

  model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

  return model

if __name__ == '__main__':
	model = get_model()
	model.summary()