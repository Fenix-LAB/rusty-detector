# Detector de corroción en metales

## Descripción
Este proyecto es un detector de corroción en metales, usando visión artificial y redes neuronales.
El objetivo es la deteccion de corrocion de metales, con un modelo de segmentacion
El entrenamiento es tranfer learning con yolov8
https://github.com/ultralytics/ultralytics


## Requerimientos
- Python

## Instalación
crear un entorno virtual
```bash
python -m venv env
```
activar el entorno virtual
```bash
env\Scripts\activate
```

instalar las librerias de python
```bash
pip install -r requirements.txt
```

instalar pytorch y tprchvision con GPU
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torchvision==0.15.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

## train
Para entrenar el modelo se usa el siguiente comando:
```bash
yolo task=segment mode=train epochs=1 data=data.yaml model=yolov8m-seg.pt imgsz=640 batch=2
```
