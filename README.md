# YOLO-RACE: ReAssembly and Convolutional Block Attention for Enhanced Dense Object Detection 

# 01. Overview of the Proposed Model
* We modified the Upsampling method in the YOLOv8 structure to CARAFE and added the ResCBAM structure to address the issue of performance degradation in detecting small objects in YOLO models.
![학위논문 그림 1](https://github.com/user-attachments/assets/78b18b04-848d-4c47-8f5a-d73edc62e4ec)

# 02. Environment
```
  pip install -r requirements.txt
```

# 03. Model Training Example
* python(start_train.py)
```
  python start_train.py --model ultralytics/cfg/models/v8/yolov8_CR.yaml --dataset_config ultralytics/cfg/datasets/SKU-110K.yaml  --epochs 100 --output_dir E:/YOLOv8_train/ --run_name YOLOv8n_SKU-110K
```
<div align=center>OR</div>

* Notebook(.ipynb)
  
```
from ultralytics import YOLO
from multiprocessing import freeze_support

# 모델 경로 설정
model_path = 'E:/Degree_project03/YOLO-RACE/ultralytics/cfg/models/v8/yolov8_CR.yaml'
# 데이터 설정 파일 경로
data_path = 'E:/Degree_project03/YOLO-RACE/ultralytics/cfg/datasets/SKU-110K.yaml'
# 학습 결과 저장 경로
project_path = 'E:/YOLOv8_train/'

# YOLOv10 모델 불러오기
model = YOLO(model_path)

if __name__ == '__main__':
    freeze_support()
    
    # 모델 학습
    model.train(data=data_path, epochs=100, project=project_path, name='yolov8n_CR_SKU10000')
```

# 04. Model Validation Example
* Notebook(.ipynb)

```
from ultralytics import YOLO

# 모델 로드
model = YOLO("E:/YOLOv8_train/yolov8n_CR_SKU/weights/best.pt")  # 사용자 정의 모델을 로드합니다

# 모델 검증
metrics = model.val()  # 인자가 필요 없음, 데이터셋과 설정이 기억됩니다
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # 각 카테고리의 map50-95가 포함된 목록
```
