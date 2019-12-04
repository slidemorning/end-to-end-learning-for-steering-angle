import keras
import parser
import utils
import numpy as np

MODEL_PATH = './steer-model/88-39.h5'
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL = 66, 200, 1

def load_model(model_path=MODEL_PATH):
    model = keras.models.load_model(model_path)
    model.summary()
    return model

# for debug
def __test__(model, X, y):
  n_data = len(X)
  for index in range(n_data):
    print('test at {} img'.format(index))
    img = X[index]
    img = np.reshape(img, (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
    print('predict value : ', model.predict(img))
    print('ground truth  : ', y[index])

if __name__ == '__main__':

    X, y = utils.load_data()
    X_test, y_test = X[9000:], y[9000:]
    args = parser.get_args()
    model = load_model(MODEL_PATH)
    __test__(model, X_test, y_test)
'''
    parse argument (input_dir, output_dir)
    load model
    load test data
    
    process
    data->preprocess->model->output->save to txt file
'''