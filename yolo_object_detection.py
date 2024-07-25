# Importing Libraries
import cv2 as cv 
import numpy as np
# from wandb import Classes
from matplotlib import pyplot as plt

# load yolo
net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg") # dnn - dense neural network
# cv.dnn.readNet(): Loads the configuration files and framework into the memory
# clasees = []
with open("coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
# print(classes)

layer_name = net.getLayerNames()
output_layer = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]
# YOLOv3 has 3 output layers  
# getLayerNames(): Get the name of all layers of the network
# getUnconnectedOutLayers(): Get the index of the output layers
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load Image
img = cv.imread("lane6.jpg")
img = cv.resize(img, None, fx=0.4, fy=0.4)
height, width, channel = img.shape

# Creation of Blob
# blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
# It returns a 4-dimensional array/blob for the input image
# scalefactor: scales down the image channel by a factor of 1/n
blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# YOLO Image Detection
net.setInput(blob)
outs = net.forward(output_layer)

# print(outs)
# Showing Information
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detection
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # cv.circle(img, (center_x, center_y), 10, (0, 255, 0), 2 )
            # Reactangle Cordinate
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# print(len(boxes))
# number_object_detection = len(boxes)
# Non Maximum Suppression selects a single entity out of many overlapping entities. 
# The criteria is usually discarding entities that are below a given probability bound.
indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3) # Confidence Threshold, Non Maximum Suppression Threshold
# print(indexes)
print(len(indexes))
detected = []
for x in indexes:
    detected.append(class_ids[x])
print(detected)

# Types of vehicles to be detected
n_person = detected.count(0) # 25 sec
n_bicycle = detected.count(1) # 20 sec
n_car = detected.count(2) # 10 sec
n_motorbike = detected.count(3) # 5 sec
n_aeroplane = detected.count(4) # 1 sec
n_bus = detected.count(5) # 15 sec
n_train = detected.count(6) # 15 sec
n_truck = detected.count(7) # 20 sec

# Time Allotment (Static)
time = n_person*25 + n_bicycle*20 + n_car*10 + n_motorbike*5 + n_aeroplane*1 + n_bus*15 + n_train*15 + n_truck*20
if(time >= 90):
    time = 90

print("no of persons = ", n_person)
print("no of bicycles = ", n_bicycle)
print("no of cars = ", n_car)
print("no of motorbikes = ", n_motorbike)
print("no of aerplanes = ", n_aeroplane)
print("no of buses = ", n_bus)
print("no of trains = ", n_train)
print("no of trucks = ", n_truck)
print("Traffic light will kept on for: ", time, " sec")

# Boxes and Labels
font = cv.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        # print(label)
        color = colors[i]
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv.putText(img, label, (x, y + 30), font, 1, color, 2)
       
# Display Image on Screen     
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
# cv.imshow("IMG", img)
# cv.waitKey(0)
# cv.destroyAllWindows()
