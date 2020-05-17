import tensorflow as tf
import numpy as np
import matplotlib as plt
import math
import time
from PIL import Image
from imageai.Detection import ObjectDetection
from tkinter import Tk
from tkinter.filedialog import askopenfilename

config = {
    "saveslices" : True,
    "drawboxes" : True,
    "saveFinalDetections" : True
}

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
filename = filename.split('/')[-1]
print(filename)

# Setting model weight
model_path = '../models/'
model_name = 'yolo.h5'
model_weight = model_path + model_name

# Configuring imageai with weight
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_weight)
detector.loadModel()

# Setting paths for inputs and outputs
input_path = '../input/images/'
output_path = '../output/images/'
result_path = '../output/result/'
file_name = filename
output_file = output_path + file_name
# print(output_file) 
result_file = result_path + file_name.split('.')[0] + '.txt'
# print(result_file)
input_file = input_path + file_name

# Reading image
image = tf.keras.preprocessing.image.load_img(input_file)   #Load image
image = tf.keras.preprocessing.image.img_to_array(image)    #Converting image to numpy array

shape = image.shape
height = shape[0]
width = shape[1]
offset_height = 0.05 * height
offset_width = 0.03 * width
vert_pieces = 2
horiz_pieces = 2
crop_height = int(math.ceil(height/vert_pieces + offset_height))
crop_width = int(math.ceil(width/horiz_pieces + offset_width))

# Calculate crop coordinates
boxes = []
x1 = 0
y1 = 0
for i in range(vert_pieces):
    y = y1 + crop_height/height
    for j in range(horiz_pieces):
        x = x1 + crop_width/width
        if(y > 1):
            y = 1
        if(x > 1):
            x = 1
        boxes.append([y1,x1,y,x])
        x1 = x - (2*offset_width)/width
    x1 = 0
    y1 = y - (2*offset_height)/height
boxes = np.array(boxes)

# Prepocessing image for tensorflow function compatibility
images = []
images.append(image)
images = np.array(images)
num_boxes = boxes.shape[0]
box_indices = np.zeros(num_boxes)

# Crop Images
crop_tensor = tf.image.crop_and_resize(images, boxes, box_indices, crop_size=[crop_height,crop_width], method='bilinear', extrapolation_value=0, name=None)
cropped_images = crop_tensor.eval(session=tf.Session())

if(config["drawboxes"]):
    draw_boxes = [boxes]
    draw_boxes = np.array(draw_boxes)
    print("drawboxes = "+str(draw_boxes.shape))
    draw_tensor = tf.image.draw_bounding_boxes(images, draw_boxes, name=None) 
    draw_img = draw_tensor.eval(session=tf.Session())
    drawn_img = tf.keras.preprocessing.image.array_to_img(draw_img[0],data_format=None, scale=True, dtype=None)
    name = output_path+file_name.split('.')[0]+"_boxes.jpg"
    drawn_img.save(name)

if(config["saveslices"]):
    for i in range(4):
        result = tf.keras.preprocessing.image.array_to_img(cropped_images[i],data_format=None, scale=True, dtype=None)
        name = output_path+file_name.split('.')[0]+"_crop"+str(i)+".jpg"
        result.save(name)

# Run detections in sliced images
tic = time.time()
crop_detections = []
for i in range(num_boxes):
    temp_img, crop_det = detector.detectObjectsFromImage(input_image=cropped_images[i],input_type="array", output_type="array",minimum_percentage_probability=50)
    crop_detections.append(crop_det)


tac = time.time()
elapsed = tac-tic
print(elapsed)

# Aggregate Detections from slices
detected_boxes = []
detections = []
scores = []
for i in range(num_boxes):
    size = len(crop_detections[i])
    x = boxes[i][1]*width
    y = boxes[i][0]*height
    for j in range(size):
        detected = crop_detections[i][j]
        box = detected['box_points']
        box = [box[0]+x, box[1]+y, box[2]+x, box[3]+y]
        detected_boxes.append(box)
        detected['box_points'] = box
        scores.append(detected['percentage_probability'])
        detections.append(detected)
detected_boxes = np.array(detected_boxes)
scores = np.array(scores)

# Apply non max suppresion to avoid overlapping detections
selected_indices = tf.image.non_max_suppression(tf.convert_to_tensor(detected_boxes, dtype=tf.float32), tf.convert_to_tensor(scores, dtype=tf.float32), len(detected_boxes), 0.3)
selected_indices = selected_indices.eval(session=tf.Session())

# Select detections not removed by NMS
final_detections = []
final_detections_boxes = []
for index in selected_indices:
    final_detections.append(detections[index])
    final_detections_boxes.append(detected_boxes[index])
len(final_detections_boxes)

print(final_detections)

#Save image with final detection boxes
if(config["saveFinalDetections"]):
    final_detections_boxes = np.array(final_detections_boxes)
    draw_final_boxes = np.zeros(final_detections_boxes.shape)
    draw_final_boxes[:,0] = final_detections_boxes[:,1]/height
    draw_final_boxes[:,1] = final_detections_boxes[:,0]/width
    draw_final_boxes[:,2] = final_detections_boxes[:,3]/height
    draw_final_boxes[:,3] = final_detections_boxes[:,2]/width
    draw_final_boxes = np.array([draw_final_boxes])
    draw_tensor = tf.image.draw_bounding_boxes(images, draw_final_boxes, name=None) 
    draw_img = draw_tensor.eval(session=tf.Session())
    drawn_img = tf.keras.preprocessing.image.array_to_img(draw_img[0],data_format=None, scale=True, dtype=None)
    name = output_path+file_name.split('.')[0]+"_final_detections.jpg"
    drawn_img.save(name)
