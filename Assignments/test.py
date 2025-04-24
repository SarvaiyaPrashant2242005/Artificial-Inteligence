import face_recognition
import cv2
import os
import numpy as np
import shutil

# Step 1: Create a ZIP of the "photos" folder
shutil.make_archive("photos", "zip", "photos")
print("✅ photos.zip created successfully!")

# Step 2: Load face encodings from the "photos" folder
photos_path = "photos/"
known_face_encodings = []
known_face_names = []

for image_name in os.listdir(photos_path):
    if image_name.endswith(".jpg") or image_name.endswith(".png"):
        image_path = os.path.join(photos_path, image_name)
        
        # Load image and encode face
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:  # If face is detected
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(image_name)[0])  # Name from file

print(f"✅ Loaded {len(known_face_names)} faces from photos folder.")

# Step 3: Initialize Webcam for Real-Time Face Recognition
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Scale back up face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw box and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

    # Show webcam feed
    cv2.imshow('Face Recognition', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
