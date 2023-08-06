import cv2
import numpy as np
import tensorflow as tf

# Load the fall detection model (replace 'model_path' with the actual path to your model file)
model_path = "C:/Users/praja/Documents/VS Python/ANN/Fall/Fall_detect_model.h5"
model = tf.keras.models.load_model(model_path)

# Function to preprocess the frames
def preprocess_frame(frame, target_size):
    frame_resized = cv2.resize(frame, target_size)
    frame_normalized = frame_resized.astype('float32') / 255.0
    return np.expand_dims(frame_normalized, axis=0)

# Function to perform fall detection on a video
def detect_fall_in_video(video_path, threshold):
    cap = cv2.VideoCapture(video_path)

    # Set the target size to match the input size expected by the model
    target_size = (256, 256)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        input_frame = preprocess_frame(frame, target_size)

        # Make a prediction using the model
        prediction = model.predict(input_frame)

        # Get the probability of "Fall" class (assuming 0 is the index of "Fall" class)
        probability_fall = prediction[0][0]

        # Apply thresholding to classify frame as "Fall" or "Non-Fall"
        is_fall = probability_fall >= threshold

        # If 'is_fall' is True, perform the action for fall detection (e.g., save frame, alert, etc.)
        if is_fall:
            label = "Fall (Probability: {:.2f})".format(probability_fall)
            color = (0, 0, 255)  # Red color for "Fall" prediction
            print("Fall Detected")
        else:
            label = "Non-Fall (Probability: {:.2f})".format(probability_fall)
            color = (0, 255, 0)  # Green color for "Non-Fall" prediction
            print("No Fall Detected")

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Fall Detection", frame)

        # Press 'q' to exit the video display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Set the threshold for fall detection (you can adjust this based on your model and requirements)
threshold = 0.5

# Specify the path to your custom video
custom_video_path = "C:/Users/praja/Documents/VS Python/ANN/Fall/queda.mp4"

# Call the fall detection function
detect_fall_in_video(custom_video_path, threshold)
