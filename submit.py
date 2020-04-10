import keras
import argparse
import os
import cv2
import numpy as np

IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL = 200, 66, 1
MODEL_PATH = './20143300_1.h5'

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str)
	parser.add_argument('--output_dir', type=str)
	args = parser.parse_args()
	return args

def load_model(model_path=MODEL_PATH):
    model = keras.models.load_model(model_path)
    model.summary()
    return model

def preprocess(image):
    image = image[50:140, 10:-10, :]
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.reshape(image, (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))

def evaluate(model, input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    f = open(output_dir+'/20143300_1.txt', 'w')
    images_path_list = os.listdir(input_dir)
    for image_path in images_path_list:
        X = cv2.imread(input_dir+'/'+image_path) # 160 320 3
        X = preprocess(X)
        yhat = model.predict(X)
        yhat = str(yhat)+'\n'
        f.write(yhat)
    f.close()


if __name__ == '__main__':
    args = get_args()
    model = load_model(MODEL_PATH)
    input_dir = args.input_dir
    output_dir = args.output_dir
    evaluate(model, input_dir, output_dir)
