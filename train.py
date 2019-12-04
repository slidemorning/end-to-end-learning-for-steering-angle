import utils
import model
import os
from keras.callbacks import ModelCheckpoint

# global var
X_PATH = './data/image/img_preprocessed_v1.h5'
Y_PATH = './data/label/label.h5'
MODEL_SAVE_FOLDER_PATH = './steer-model/'

# hyperparameter of steer model
EPOCHS = 100
BATCH_SIZE = 64

if __name__ == '__main__':

    X, y = utils.load_data(X_PATH, Y_PATH)

    X, y = utils.shuffle_data(X, y)

    X_train, X_valid, X_test = X[:7000], X[7000:9000], X[9000:]

    y_train, y_valid, y_test = y[:7000], y[7000:9000], y[9000:]

    steer_model = model.get_model()
    steer_model.summary()

    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)
    model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:04d}-{val_loss:.4f}.h5'
    cb_checkpoint = ModelCheckpoint(filepath=model_path,
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True)

    history = steer_model.fit(X_train,
                              y_train,
                              epochs=EPOCHS,
                              batch_size=BATCH_SIZE,
                              validation_data=(X_valid, y_valid),
                              verbose=1,
                              callbacks=[cb_checkpoint])

    utils.plot_history(history)

    #steer_model.save(MODEL_SAVE_PATH+MODEL_SAVE_NAME)

