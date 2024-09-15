from src import utils
from PIL import Image

def evaluate_model(model, val_loader, cfg, device):
    logger = utils.setup_logger(__name__)

    # Set up validation arguments
    args = dict(
        data=cfg['data_dir'],
        batch=cfg['batch_size'],
        # imgsz=cfg['img_size'],
        device=cfg['device'],
        workers=cfg['num_workers'],
    )

    # Validate the model
    results = model.val(**args)

    map50 = results.maps[0]
    logger.info(f"mAP@0.5: {map50:.4f}")

    return map50

def run_inference(model, image_path, cfg, device):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    
    # Run inference
    results = model(image, conf=cfg['conf_thres'], iou=cfg['iou_thres'])

    return results  # Return the first (and only) result
