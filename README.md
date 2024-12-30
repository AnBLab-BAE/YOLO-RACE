# YOLO-RACE: ReAssembly and Convolutional Block Attention for Enhanced Dense Object Detection 

# 1. Overview of the Proposed Model
* We modified the Upsampling method in the YOLOv8 structure to CARAFE and added the ResCBAM structure to address the issue of performance degradation in detecting small objects in YOLO models.
![학위논문 그림 1](https://github.com/user-attachments/assets/78b18b04-848d-4c47-8f5a-d73edc62e4ec)

# 2. Model Train 
* start_train.py
```
  python start_train.py --model ultralytics/cfg/models/v8/yolov8_CR.yaml --dataset_config ultralytics/cfg/datasets/SKU-110K.yaml  --epochs 100 --output_dir E:/YOLOv8_train/ --run_name YOLOv8n_SKU-110K
```



# 3. Model Validation 
