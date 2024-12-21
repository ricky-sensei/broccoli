import os
import shutil
from ultralytics import YOLO
from update_path import update_path

update_path()

# detectディレクトリがあった場合は削除して再作成
detect_dir = "runs/detect"
if os.path.exists(detect_dir):
    for dir in os.listdir(detect_dir):
        if "predict" in dir:
            shutil.rmtree(f"{detect_dir}/{dir}")
if os.path.exists("results"):
    shutil.rmtree("results")
    os.makedirs("results")
    

# inference_dataの中のファイルを推論
file_names = os.listdir( "inference_data")
for file_name in file_names:
    source = f"inference_data/{file_name}"
    print(source)
    model = YOLO("../test_labelImg/runs/detect/train/weights/best.pt")
    model.predict(source, save=True, imgsz=640, conf=0.5, )

# 出力結果をresultsディレクトリにコピー
result_dirs = os.listdir(detect_dir)
print(result_dirs)
for dir in result_dirs:
    file= os.listdir(f"{detect_dir}/{dir}")[0]
    filepath = f"{detect_dir}/{dir}/{file}"
    shutil.copy(filepath, f"./results")

print("Done")
