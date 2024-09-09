import cv2

from deepface import DeepFace

# Access webcam

cap = cv2.VideoCapture(0)

# Read and process frames continuously

while True:

  # Capture frame-by-frame

  ret, frame = cap.read()

  if not ret:

    break

  # Perform emotion detection

  result = DeepFace.analyze(frame, actions=['emotion'],enforce_detection=False)

  print(result[0]['dominant_emotion'])

  # Display the frame

  cv2.imshow('Webcam', frame)

  # Press 'q' to exit the loop

  if cv2.waitKey(1) & 0xFF == ord('i'):

    break

# Release the webcam and close all OpenCV windows q

cap.release()

cv2.destroyAllWindows()