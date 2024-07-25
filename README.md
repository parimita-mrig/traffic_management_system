# Traffic-Management-by-Vehicle-Detection

## Problem Statement
To build a self adaptive traffic light control system based on YOLO. Disproportionate and diverse traffic in different lanes leads to inefficient utilization of same time slot for each of them characterized by slower speeds, longer trip times, and increased vehicular queuing.
To create a system which enable the traffic management system to take time allocation decisions for a particular lane according to the traffic density on other different lanes with the help of cameras, image processing modules. 

## About the Implementation
Monitor traffic on a crossroad using cameras, each monitoring a lane.     
These cameras will input images every few seconds to the software. Different types of vehicles will be detected using YOLO.    
Based on this data the traffic lights will be set for an adequate time.    
This will lead to more rate of traffic flow, through the signals.    
 
## Prerequisite
Image Input - Using real time traffic images.     
Object Detecion - YOLO v3 model is used for image detection.    

Numpy - For handling arrays.    
Matplotlib - To display the results in a window.    
Open CV - To load and use the YOLO v3 model.    
YOLO v3 - For object detection. Weights, configuration and coco name files are downlaoded from the YOLO v3 website.     

## About YOLO
Advantages -    
Speed (On a Pascal Titan X, YOLO v3 processes images at 30 FPS and has a mAP of 57.9% on COCO test-dev).    
The network is able to generalize the image better.     
Faster version (with smaller architecture)       
Open Source: https://pjreddie.com/darknet/yolo/    

Limitation -      
The model struggles with small objects that appear in groups.     

## Acknowledgements 
Resources:    
Blog -     
https://medium.com/@MrBam44/yolo-object-detection-using-opencv-with-python-b6386c3d6fc1      
Yolo -      
https://pjreddie.com/darknet/yolo/ (Download YOLO weights from here.)     
Research paper -    
https://www.atlantis-press.com/journals/ijcis/125941268/view     
