import numpy as np
import random
import h5py
import cv2


'''
Keras Conv2D parameter
data_format = 'channel_last' correspond to input with shape (batch, height, widht, channels)
            or 'channel_first' correspond to input with shape(batch, channels, height, widht)
'''


# img   : 52722  (20Hz)
# label : 263583 (100Hz)
X_PATH = './data/image/image.h5'
Y_PATH = './data/label/label.h5' 
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
    print(img_set[rand_num].shape)
    cv2.imshow('img', img_set[rand_num])
    if cv2.waitKey(0) == 27:
        return rand_num

def show_random_crop_img(img_set):
	rand_num = random.randint(1, 10000)
	print('image at {}'.format(rand_num))
	img = img_set[rand_num]
	img = img[50:140, :, :]
	print(img.shape)
	cv2.imshow('crop img', img)
	if cv2.waitKey(0) == 27:
		return rand_num

def crop(img):
	return img[50:140, :, :]

def resize(img):
	# resize to (width, height)
	return cv2.resize(img, (200, 66), cv2.INTER_AREA)

def show(img):
	cv2.imshow('img', img)
	cv2.waitKey(0)

def preprocess(img):
	print(img)
	print(img.shape)
	img = img[50:140, :, :]
	print(img)
	print(img.shape)
	#img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
	return img


if __name__ == '__main__':
	X, y = load_data()
	img = X[1000]
	img = crop(img)
	img = resize(img)
	show(img)