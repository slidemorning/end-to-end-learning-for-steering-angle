from model import get_model
import utils


if __name__ == '__main__':
    model = get_model()
    model.summary()

    X, y = utils.load_data()
    print(X.shape)
    print(y.shape)
