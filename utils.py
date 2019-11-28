import numpy as np
import random
import h5py
import cv2


# img   : 52722  (20Hz)
# label : 263583 (100Hz)
X_PATH = './data/image/image.h5' # args : input dir
Y_PATH = './data/label/label.h5' # args : input dir
TARGET_KEY_X = 'X'
TARGET_KEY_Y = 'steering_angle'
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# internal
def _down_sampling(data, hz_src, hz_dest):
    ret = []
    inc = hz_src//hz_dest
    for i in range(inc*10000, inc*20000, inc):
        ret.append(data[i])
    return np.array(ret)

def load_data(x_path=X_PATH, y_path=Y_PATH):
    h5_x = h5py.File(x_path, 'r')
    h5_y = h5py.File(y_path, 'r')
    X = h5_x[TARGET_KEY_X][10000:20000]
    X = np.moveaxis(X, 1, -1)
    y = h5_y[TARGET_KEY_Y]
    y = _down_sampling(y, 100, 20)
    return X, y

# for debug
def show_random_img(img_set):
    rand_num = random.randint(1, 10000)
    print('image at {}'.format(rand_num))
    cv2.imshow('img', img_set[rand_num])
    if cv2.waitKey(0) == 27:
        return rand_num

def preprocess(img):
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return img


if __name__ == '__main__':

    X, y = load_data(X_PATH, Y_PATH)
    print(X.shape)