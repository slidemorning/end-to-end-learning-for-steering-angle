import numpy as np
import random
import h5py
import cv2


# img   : 52722  (20Hz)
# label : 263583 (100Hz)
X_PATH = './data/image/image.h5'
Y_PATH = './data/label/label.h5'
TARGET_KEY_X = 'X'
TARGET_KEY_Y = 'steering_angle'

# internal
def _down_sampling(data, hz_src, hz_dest):
    # down-sampling : from 100hz label to 20hz label
    ret = []
    inc = hz_src//hz_dest
    for i in range(inc*10000, inc*20000, inc):
        ret.append(data[i])
    return np.array(ret)


def load_data(x_path, y_path):
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

if __name__ == '__main__':

    X, y = load_data(X_PATH, Y_PATH)
    #rand_idx = show_random_img(X)
    #print('label : ', y[rand_idx])
    print(y.shape)
    for val in y:
        print(val)