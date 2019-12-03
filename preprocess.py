import numpy as np
import h5py
import cv2

READ_PATH = './data/image/image.h5'
SAVE_PATH = './data/image/'
IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL = 200, 66, 1
KEY = 'X'

def __load_origin_data__(path=READ_PATH):
    h5 = h5py.File(path, 'r')
    X = h5['X']
    X = X[10000:20000]
    return X

def __preprocess__(data, hwc=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)):
    ret = np.empty([len(data), hwc[0], hwc[1], hwc[2]])
    data = data[:, :, 50:140, 10:-10]
    #print(data.shape)
    data = np.moveaxis(data, 1, -1)
    #print(data.shape)
    for index in range(len(data)):
        img = data[index]
        #print(img.shape)
        img = cv2.resize(img, (hwc[1], hwc[0]), cv2.INTER_AREA)
        #print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.reshape(img, (hwc[0], hwc[1], hwc[2]))
        ret[index] = img
    return ret

def save_preprocessed_data(r_path, s_path, s_name):
    h5 = h5py.File(s_path+s_name, 'w')
    X = __load_origin_data__(r_path)
    X = __preprocess__(X, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
    h5.create_dataset(KEY, data=X)
    h5.close()

if __name__ == '__main__':

    save_file_name = 'image_preprocessed.h5'
    #save_preprocessed_data(READ_PATH, SAVE_PATH, save_file_name)
    #X = load_origin_data(READ_PATH)

    # h5_origin = h5py.File(READ_PATH, 'r')
    # h5_preprocessed = h5py.File(SAVE_PATH+SAVE_NAME, 'w')
    #
    # X_origin = h5_origin['X']
    # X_origin = X_origin[10000:20000]
    # print('preprocessing data ...')
    # X_preprocessed = preprocess(X_origin, (66, 200, 1))
    # print('complete prerpocess ...')
    # print('save hdf5 file')
    # h5_preprocessed.create_dataset('X', data=X_preprocessed)
    # h5_preprocessed.close()

    # h5 = h5py.File('./data/image/img_preprocessed_v1.h5', 'r')
    # X = h5['X']
    # print(X.shape)