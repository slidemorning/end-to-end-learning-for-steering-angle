from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D

IMG_CH, IMG_ROW, IMG_COL = 3, 160, 320

def get_model():
	model = Sequential()
	model.add(Lambda(lambda x: x/127.5-1.0,
	input_shape=(IMG_CH, IMG_ROW, IMG_COL),
	output_shape=(IMG_CH, IMG_ROW, IMG_COL)))
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same'))
	model.add(ELU())
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='same'))
	model.add(ELU())
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='same'))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(512))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1))

	model.compile(optimizer='adam', loss='mse')

	return model

if __name__ == '__main__':
	print('model.py test')
	model = get_model()
	model.summary()
