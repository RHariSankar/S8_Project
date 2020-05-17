import cv2
import math


def drawBoxes(input_path,output_path,detections):

    image = cv2.imread(input_path) 
    for box in detections:
        x1,y1,x2,y2 = (box['box_points'][0], box['box_points'][1], box['box_points'][2], box['box_points'][3])
        confidence = box['percentage_probability']
        label = box['name']
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        text = label+" "+str(round(confidence,2))
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
        labelSize=cv2.getTextSize(text,cv2.FONT_HERSHEY_COMPLEX,0.5,2)
        _x1 = x1
        _y1 = y1
        _x2 = _x1+labelSize[0][0]
        _y2 = y1-int(labelSize[0][1])
        cv2.rectangle(image,(_x1,_y1),(_x2,_y2),(0,255,0),cv2.FILLED)
        cv2.putText(image,text,(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
        # print("Saving in "+ output_path)
        # out = output_path + "/result.jpg"
        cv2.imwrite(output_path,image)