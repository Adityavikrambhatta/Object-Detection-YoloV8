from ultralytics import YOLO

def create_model(cfg):
    # Load a pre-trained YOLOv8 model
    model = YOLO(cfg['model_name'])

    # Customize the model for our specific number of classes
    model.model.model[-1].nc = cfg['num_classes']
    # model.names = cfg['class_names']

    return model

def best_model(cfg):

    # Providing the best model to get evaluated
    model = YOLO(cfg['model_name'])
    model.model.model[-1].nc = cfg['num_classes']
    model = YOLO(cfg['best_model'])
    return model

def get_model_device(model):
    return next(model.parameters()).device

def model_to_device(model, device):
    model.to(device)
    return model
