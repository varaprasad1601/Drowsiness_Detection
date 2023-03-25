import cv2

cap = cv2.VideoCapture(0)
eye_cascade = cv2.CascadeClassifier("./haarcascade/fulleye.xml")

i=1
while True:
    ret, roi_color = cap.read()
    eye = eye_cascade.detectMultiScale(roi_color, 1.1, 3)
    for (ex, ey, ew, eh) in eye:
        roi_gray = roi_color[ey:ey+eh, ex:ex+ew]
        cv2.imwrite("./pics/img"+str(i)+".png",roi_gray)
        i+=1
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        cv2.imshow("roi_frames",roi_color)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

