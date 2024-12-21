from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data="dataset.yaml", epochs=100, batch=8)
