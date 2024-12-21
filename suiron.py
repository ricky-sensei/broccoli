from ultralytics import YOLO

source = "./suiron_test.jpg" # 自身が検出したいデータの位置
model = YOLO("../test_labelImg/runs/detect/train/weights/best.pt") # 学習した重みデータ
model.predict(source, save=True, imgsz=640, conf=0.5)
