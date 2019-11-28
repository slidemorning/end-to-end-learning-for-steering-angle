from keras.layers import Dense, Activation, Flatten, Conv2D, Lambda
from keras.layers import MaxPooling2D, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import utils

def get_model(image_x, image_y):

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(image_x, image_y, 3)))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Dense(256))
    model.add(Dense(64))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    filepath = "20143300.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    callback_list = [checkpoint]

    return model, callback_list

if __name__ == '__main__':

    X, y = utils.load_data()
    X_train, X_valid = X[:7000], X[7000:]
    y_train, y_valid = y[:7000], y[7000:]
    model, callbacks_list = get_model(320, 160)
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=3, batch_size=100, callbacks=callbacks_list)
    print(model.summary())

