import os
import cv2
import time
import math
import wave
import pyaudio
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


face_cascade = cv2.CascadeClassifier("./haarcascade/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("./haarcascade/fulleye.xml")
model = load_model("./dnn_model.h5")
print(model)
faces = []
eyes = []

#------------------------------------ Face Detect Fuction --------------------------------
def detect_face(frame, i):
    face = face_cascade.detectMultiScale(frame, 1.3, 5)

    if len(face)>0:
        faces.append(1)
    else:
        faces.append(0)
        
    for (x, y, w, h) in face:
        
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Extract the region of interest (ROI) which is the face area
            roi_color = frame[y:y+h, x:x+w]
            cv2.imshow("roi",roi_color)

            # Detect eyes and extract
            eye = eye_cascade.detectMultiScale(roi_color, 1.1, 3)
            eyes = detect_eyes(roi_color, eye, w, i)
    i+=1
            
    return faces, face







#------------------------------------ Detect Eyes and Predict Fuction --------------------------------
def detect_eyes(roi_color, eye, w, i):
    # Iterate over each detected eye
    for (ex, ey, ew, eh) in eye:
        # Draw a rectangle around the eye
        #cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Check if the eye is on the left or right side of the face
        roi_gray = roi_color[ey:ey+eh, ex:ex+ew]

        cv2.imshow("roi_frame",roi_gray)
        

        #intensity_matrix = np.ones(roi_gray.shape, dtype = "uint8")*50

        #roi_bright = cv2.add(roi_gray,intensity_matrix)
        #rroi_bright = cv2.add(roi_gray,intensity_matrix)
        #cv2.imwrite("./pics/img"+str(i)+".png",roi_gray)
        #rroi_bright = cv2.resize(rroi_bright,(500,500))
        cv2.imshow("roi_bright",roi_gray)

        # Resize the face region to match the input shape of the model
        roi_gray = cv2.resize(roi_gray, (64, 64))
        # Convert the face region to a numpy array
        roi_array = img_to_array(roi_gray)
        # Reshape the numpy array to match the input shape of the model
        roi_array = roi_array.reshape(1, 64, 64, 1)
        # Make a prediction using the model
        prediction = model.predict(roi_array)
        # Get the predicted class index
        predicted_class = np.argmax(prediction)
        print(predicted_class)

        # Draw a rectangle around the detected face
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        # Draw a label for the predicted class on the frame
        label = "Open Eyes" if predicted_class == 0 else "Closed Eyes"
        cv2.putText(roi_color, label, (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2)








#------------------------------------- Alert function --------------------------------
def play_alert():
    # set the file name and open the file
    filename = "./alarm.wav"
    wf = wave.open(filename, 'rb')

    # instantiate PyAudio
    p = pyaudio.PyAudio()

    # open a stream
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # read data and play audio
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    # close the stream and terminate PyAudio
    stream.close()
    p.terminate()

    return True







#------------------------------------- Time Check function --------------------------------
def time_check(start_time, obj_list, alert_count, sec, fps):
    br = False
    alert = False
    obj_list_range = obj_list[-11:]
    res = max(set(obj_list_range), key = obj_list_range.count)
    elapsed_time = time.time() - start_time
    #print("start_time :",start_time,"---------> elapsed_time :",elapsed_time)

    print(obj_list_range)
    #print("res :",res)
    if int(elapsed_time%sec) == 9:
        if res == 0:
            sec = 15
            alert = play_alert()
            if alert:
                alert_count +=1
                print("alert_alert:",alert)
                print("alert_count:",alert_count)
                obj_list.clear()
                if alert_count == 5:
                    br = True
                    os.system('rundll32.exe powrprof.dll,SetSuspendState 0,1,0')
            else:
                pass
        else:
            sec = 15
            print("alert_alert:",alert)
            alert_count = 0
    return res, alert_count, br, sec
    

            
                




#--------------------------------------- Main Function --------------------------------
def main():
    alert_count = 0
    i=0
    sec = 10
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = math.ceil(fps)
    while True:
        i+=1
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_list, face = detect_face(frame, i)

        if len(face)==0:
            face_time_check, alert_count, br, sec = time_check(start_time,faces_list,alert_count, sec, fps)
            if br:
                break
        else:
            start_time = time.time()

          
        # Show Captured video
        cv2.imshow("frame",frame)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
main()
