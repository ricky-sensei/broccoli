import os
from ultralytics import YOLO
from update_path import update_path

update_path()

dir = "inference_data"
file_names = os.listdir(dir)

for file_name in file_names:
    source = f"{dir}/{file_name}"
    model = YOLO("../test_labelImg/runs/detect/train/weights/best.pt")
    model.predict(source, save=True, imgsz=640, conf=0.5)
