# Importamos las librerias
from ultralytics import YOLO
import cv2

# load model
model = YOLO("model.pt")

# init camera
cap = cv2.VideoCapture(0) # Para cambiar la camara

# loop
while True:
    # read frames
    ret, frame = cap.read()

    # resultados use the model for detection and segmentation
    resultados = model.predict(frame, imgsz = 640, conf = 0.7) # 0

    # Confidence conf = 0.7 X
    # conf = 0.9   OK

    # creat a new frame with the results
    anotaciones = resultados[0].plot()

    # show the frame
    cv2.imshow("DETECCION DE CORROCION Y SEGMENTACION", anotaciones)

    # break if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()