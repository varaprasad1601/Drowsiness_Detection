import cv2
import numpy as np
import math



def user_login():
    # Initialize the video capture
    cap = cv2.VideoCapture(0)

    # Create a face detector
    face_cascade = cv2.CascadeClassifier("./haarcascade/haarcascade_frontalface_default.xml")

    # Initialize a dictionary to store the known face encodings
    known_faces = {}
    names_list = []

    # Load the known face encodings from the file
    with open("known_faces.txt", "r") as f:
        for line in f:
            # Split the line into the name and face encoding
            name, face_encoding = line.strip().split(",", 1)
            # Convert the face encoding from a string to a numpy array
            face_encoding = np.array(eval(face_encoding))
            # Add the face encoding to the dictionary
            known_faces[name] = face_encoding
            names_list.append(name)

    print(names_list)

    # Capture frames and compute face encodings for the user
    face_encodings = []
    count = 0


    # Frames
    frames = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    print(frames)
    count=0


    while True:
        ret, frame = cap.read()
        # Capture 10 images of the user's face
        if count <= 10:
            # Capture a frame from the video feed
            if (count*frames)%frames==0:
                print(count)
                count+=1
                # Convert the frame from BGR color to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the grayscale frame
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                # Loop through each face found in the frame
                for (x, y, w, h) in faces:
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



    user_count = 0
    user_name = ""
    Login = "False"

    for user in names_list:
    # Compare the computed face encoding with the stored face encodings for the user
        if user in known_faces:
            distances = np.linalg.norm(face_encodings - known_faces[user], axis=1)
            min_distance = np.min(distances)

            # Check if the minimum distance is less than a threshold
            if min_distance < 20:
                user_count = 1
                user_name = user
                print("Accessed")
                break
            else:
                user_count = 0
                # If the distance is greater than or equal to the threshold, the user is not authenticated
                print("Access denied")

    if user_count > 0:
        login = "True"
        # If the distance is less than the threshold, the user is authenticated
        print("Login Successfully\nWelcome, " + user_name)
        
    else:
        login = "False"
        print("Login Failed!!!")

    cap.release()
    cv2.destroyAllWindows()

    return user_name, login
            
