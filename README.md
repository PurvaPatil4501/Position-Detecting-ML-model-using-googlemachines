# Position-Detecting-ML-model-using-googlemachines
This is Python code for the machine learning model for positions detecting machine learning model. This model detects the human postures like sitting,standing and sleeping. I hope this repository will help you with the code. The further modifications might also be done by adding new postures or classes.

# importing the essential libraries
import cv2
import numpy as np
import tensorflow as tf
import json
from teachablemachinepose import TMPose  (This is for actual importing of model)
# link - https://teachablemachine.withgoogle.com/models/HglBwRMSW/

# Load the model
url = "https://teachablemachine.withgoogle.com/models/HglBwRMSW/"
model = TMPose.load(url+"model.json", url+"metadata.json")

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Resize the frame
    frame = cv2.resize(frame, (200, 200))

    # Convert the frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Normalize the frame
    frame = np.expand_dims(frame / 255., axis=0)

    # Make a prediction
    predictions = model.predict(frame)

    # Draw the pose and labels
    label_container = ""
    for i, prediction in enumerate(predictions[0]):
        class_prediction = f"{prediction['className']}: {prediction['probability']:.2f}"
        label_container += f"<div>{class_prediction}</div>"

    # Draw the pose
    pose = model.estimate_pose(frame)
    if pose is not None:
        TMPose.draw_pose(pose, frame)

    # Display the frame
    cv2.imshow("Teachable Machine Pose Model", frame)

    # Show the labels
    cv2.createTrackbar("Labels", "Teachable Machine Pose Model", 0, 1, lambda x: None)
    if cv2.getTrackbarPos("Labels", "Teachable Machine Pose Model") == 1:
        cv2.imshow("Labels", np.array(label_container))

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
