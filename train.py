import utils
import model

### require : model check point

# global var
X_PATH = './drive/My Drive/img_preprocessed_v1.h5'
Y_PATH = './drive/My Drive/label.h5'
MODEL_SAVE_PATH = './steer-model/'
MODEL_SAVE_NAME = 'v1_100.h5' # {X_data_ver}_{epochs}.h5

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

    history = steer_model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_valid, y_valid),
                        verbose=1)

    utils.plot_history(history)

    steer_model.save(MODEL_SAVE_PATH+MODEL_SAVE_NAME)

