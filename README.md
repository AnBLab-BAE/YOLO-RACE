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

# Set the model path
model_path = 'E:/Degree_project03/YOLO-RACE/ultralytics/cfg/models/v8/yolov8_CR.yaml'

# Set the data configuration file path
data_path = 'E:/Degree_project03/YOLO-RACE/ultralytics/cfg/datasets/SKU-110K.yaml'

# Set the path to save training results
project_path = 'E:/YOLOv8_train/'

# Load the model
model = YOLO(model_path)

if __name__ == '__main__':
    freeze_support()
    
     # Train the model
    model.train(data=data_path, epochs=100, project=project_path, name='yolov8n_CR_SKU10000')
```

# 04. Model Validation Example
* Notebook(.ipynb)

```
from ultralytics import YOLO

# Load the model
model = YOLO("E:/YOLOv8_train/yolov8n_CR_SKU/weights/best.pt")  # Load Custom Model

# 모델 검증
metrics = model.val()  # No arguments needed, the dataset and settings are retained
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # List of mAP50-95 values for each category
```
