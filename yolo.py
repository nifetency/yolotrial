import cv2
import numpy as np

#Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

classes = []
with open("coco.names","r") as f :
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size = (len(classes),3))

#Loading image
img = cv2.imread("download (1).jpg")
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0), True, crop=False)


net.setInput(blob)
outs = net.forward(output_layers)


#Show info on screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            centre_x = int(detection[0]*width)
            centre_y = int(detection[1]*height)
            w = int(detection[2]* width)
            h =  int(detection[3]*height)


            #Rectangle coordinates
            x = int(centre_x - w / 2)
            y = int(centre_y - w / 2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)


indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h = boxes[i]
        label = classes[class_ids[i]]
        color = colors[i]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,label,(x,y+30),font,1,(0,0,0),3)




cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()