import cv2
import numpy as np
import math



def user_register():
    # Initialize the video capture
    cap = cv2.VideoCapture(0)

    # Ask the user for their name
    name = input("Enter your name: ")
    mobile = input("Enter your Mobile Number: ")

    name_mobile = name+"-"+mobile

    print(name_mobile)

    # Create a face detector
    face_cascade = cv2.CascadeClassifier("./haarcascade/haarcascade_frontalface_default.xml")

    # Initialize a list to store the face encodings
    face_encodings = []


    # Frames
    frames = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    print(frames)
    count=0


    while True:
        ret, frame = cap.read()
        # Capture 20 images of the user's face
        if count <= 60:
            # Capture a frame from the video feed
            if (count*frames)%frames==0:
                print(count)
                # Convert the frame from BGR color to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the grayscale frame
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                # Loop through each face found in the frame
                for (x, y, w, h) in faces:
                    count+=1
                    # Extract the face region of interest (ROI)
                    face_roi = gray[y:y+h, x:x+w]

                    # Resize the face ROI to a fixed size
                    face_roi = cv2.resize(face_roi, (128, 128))

                    # Normalize the face ROI pixel values to be between 0 and 1
                    face_roi = face_roi / 255.0

                    # Reshape the face ROI into a 1D numpy array
                    face_encoding = face_roi.reshape(1, -1)

                    # Append the face encoding to the list of face encodings
                    face_encodings.append(face_encoding)

                    # Draw a rectangle around the face in the original frame
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Display the frame
                cv2.imshow('Video', frame)

                # Exit the program if the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
        else:
            break

    # Convert the list of face encodings to a numpy array
    face_encodings = np.concatenate(face_encodings)

    # Save the face encoding with the user's name to a file
    with open("known_faces.txt", "a") as f:
        f.write(name_mobile + "," + str(face_encoding.tolist()) + "\n")

    # Train a model using the face encodings
    # ...
    # Add code to train the model here

    # Display a message to the user that their face has been registered and the model has been trained
    print("Face registered and model trained")

    # Exit the program
    cap.release()
    cv2.destroyAllWindows()
    quit()
