from ultralytics import YOLO
import cv2

"""
This a simple script uses the YOLO model to detect rust in real time using the webcam.

"""

# load model
model = YOLO("model.pt")
# init camera
cap = cv2.VideoCapture(0) # Para cambiar la camara

# loop
while True:
    # read frames
    ret, frame = cap.read()

    # resultados use the model for detection and segmentation
    result = model.predict(frame, imgsz = 640, conf = 0.7) 

    # creat a new frame with the results
    detection = result[0].plot()

    # show the frame
    cv2.imshow("Rusty Detection", detection)

    # break if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()