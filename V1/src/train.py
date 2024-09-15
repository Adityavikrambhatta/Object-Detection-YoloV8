from ultralytics import YOLO
from src import utils
from src import config
def train_model(model, train_loader, val_loader, cfg, device):
    logger = utils.setup_logger(__name__)

    # Set up training arguments
    args = dict(
        data=cfg['data_dir'],
        epochs=cfg['num_epochs'],
        batch=cfg['batch_size'],
        # imgsz=cfg['img_size'],
        device=cfg['device'],
        workers=cfg['num_workers'],
    )

    # Train the model
    results = model.train(**args)

    # Log the results
    logger.info(f"Training completed. Final mAP@0.5: {results.maps[5]:.4f}")

    return model

def validate_model(model, val_loader, cfg, device):
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

    return results.maps[50]  # Return mAP@0.5
