import yaml

def get_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Default configuration 
default_config = {
    'data_dir': 'data.yaml',
    'data' : 'data_set',
    'annotations_file': 'data_set/annotations/bbox-annotations.json',
    'batch_size': 16,
    'num_workers': 4,
    'learning_rate': 0.01,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'num_epochs': 100,
    # 'img_size': 640,
    'conf_thres': 0.25,
    'iou_thres': 0.45,
    'device': 'cpu',
    'model_name': 'yolov8n.pt',
    'num_classes': 2,
    'class_names': {0 : "person", 1 : "car"}
}