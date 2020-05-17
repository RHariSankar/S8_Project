import tensorflow as tf
import numpy as np
import matplotlib as plt
import math
import time
from PIL import Image
from imageai.Detection import ObjectDetection

from detection import *

# Setting model weight
model_path = '../models/'
model_name = 'yolo.h5'
model_weight = model_path + model_name

# Configuring imageai with weight
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_weight)
detector.loadModel()

input_path = '../input/images/'
output_path = '../output/images/'
result_path = '../output/result/'
file_name = "crowd.jpg"
output_file = output_path + file_name
# print(output_file) 
result_file = result_path + file_name.split('.')[0] + '.txt'
# print(result_file)
input_file = input_path + file_name

# Reading image
image = tf.keras.preprocessing.image.load_img(input_file)   #Load image
image = tf.keras.preprocessing.image.img_to_array(image)    #Converting image to numpy array

result = Detection(detector,image,0.05,0.03,2,2)
print(result)
print(len(result))