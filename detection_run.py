import cv2
import numpy as np
import os
from playsound import playsound
import threading



# Paths to the model files in this code the files are in the same directory as the code
proto_file = 'deploy.prototxt'
model_file = 'mobilenet_iter_73000.caffemodel'


# Sound file for alert
alert_sound = '</path to alert sound>/Alert.mp3'
alert_sound_playing = False
alert_sound_play_count=0

# Check if files exist
if not os.path.exists(proto_file) or not os.path.exists(model_file):
    raise FileNotFoundError("Required files not found: {}, {}".format(proto_file, model_file))

if not os.path.exists(alert_sound):
    raise FileNotFoundError("Alert sound file not found: {}".format(alert_sound))

# Load the COCO class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Load the pre-trained MobileNet SSD model and the corresponding weights
net = cv2.dnn.readNetFromCaffe(proto_file, model_file)

# Initialize video capture (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Set camera parameters if needed (e.g., frame width and height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define a constant factor for distance calculation
constant_factor = 45

prev_box = None
def smooth_box(prev_box, current_box, alpha=1.0):
    if prev_box is None:
        return current_box
    return (1 - alpha) * np.array(prev_box) + alpha * np.array(current_box)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to a fixed size (e.g., 300x300) for the model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Pass the blob through the network and obtain the detections
    net.setInput(blob)
    detections = net.forward()

    # Initialize flag to check if any vehicle is detected too close
    vehicle_detected_close = False

    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than a minimum threshold
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] in ["car", "bus", "motorbike", "person"]:
                # Compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Smoothing the bounding box
                if prev_box is not None:
                    smoothed_box = smooth_box(prev_box, (startX, startY, endX, endY))
                    (startX, startY, endX, endY) = smoothed_box.astype("int")

                # Check if the detected vehicle is too close (e.g., taking up more than a certain percentage of the frame)
                box_area = (endX - startX) * (endY - startY)
                frame_area = frame.shape[0] * frame.shape[1]
                if box_area / frame_area > 0.5:  # Adjust this threshold as needed
                    vehicle_detected_close = True

                # Draw the bounding box and label on the frame
                object_width = endX - startX
                distance = constant_factor / object_width
                # Convert the distance from meters to centimeters
                distance_cm = distance * 100
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, "Distance: {:.2f}m".format(distance), (10, startY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Update previous box
                prev_box = (startX, startY, endX, endY)
    # Flip the frame horizontally
    #frame = cv2.flip(frame, 1)

    # Display the resulting frame
    cv2.imshow("Vehicle Detection", frame)

    # Play sound if a vehicle is detected too close
    if cv2.waitKey(1) & 0xFF == ord ('o'):
        if vehicle_detected_close:        
            cv2.putText(frame, "Do Not Open The Door, ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if not alert_sound_playing:
                
                if alert_sound_play_count <3:
                    threading.Thread(target=playsound, args=(alert_sound,)).start()                 
                    alert_sound_playing = True 
                    alert_sound_play_count += 1        
            else:
                alert_sound_playing = False

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()