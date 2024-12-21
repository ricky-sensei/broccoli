from ultralytics import YOLO
from update_path import update_path

update_path()

model = YOLO('yolov8n.pt')
model.train(data="dataset.yaml", epochs=100, batch=8)
