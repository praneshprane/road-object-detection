import cv2
import numpy as np

# Load YOLO pre-trained model and classes
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set the input image and resizing parameters
image = cv2.imread("road_image.jpg")
width, height = 416, 416  # YOLO requires input image size to be a multiple of 32
scale = 0.00392

# Create a blob from the image and pass it through the network
blob = cv2.dnn.blobFromImage(image, scale, (width, height), (0, 0, 0), True, crop=False)
net.setInput(blob)

# Obtain the output layer names and forward pass the blob through the network
output_layers = net.getUnconnectedOutLayersNames()
layer_outputs = net.forward(output_layers)

# Process each output layer
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5 and classes[class_id] == "car":
            # Extract object coordinates and draw bounding box
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            bbox_width = int(detection[2] * width)
            bbox_height = int(detection[3] * height)

            x = int(center_x - bbox_width / 2)
            y = int(center_y - bbox_height / 2)

            cv2.rectangle(image, (x, y), (x + bbox_width, y + bbox_height), (0, 255, 0), 2)
            cv2.putText(image, "Car", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the output image
cv2.imshow("Road Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
