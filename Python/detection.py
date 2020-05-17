import math
import tensorflow as tf
import numpy as np

def Detection(detector,image,height_offset,width_offset,vertical_pieces,horizontal_pieces):

    shape = image.shape
    height = shape[0]
    width = shape[1]
    offset_height = height_offset * height
    offset_width = width_offset * width
    vert_pieces = vertical_pieces
    horiz_pieces = horizontal_pieces
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

    crop_detections = []
    for i in range(num_boxes):
        temp_img, crop_det = detector.detectObjectsFromImage(input_image=cropped_images[i],input_type="array", output_type="array",minimum_percentage_probability=50)
        crop_detections.append(crop_det)

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
    len(final_detections_boxes)

    return final_detections