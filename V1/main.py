import argparse
import torch
from src import config, dataset, model, train, evaluate, utils

def main(args):
    # Set up logging
    logger = utils.setup_logger(__name__)

    # Load configuration
    cfg = config.get_config(args.config)

    # # Dataset Preparation 
    train_set, valid_set = dataset.create_datasets(cfg, args.mode)

    # Initialize model
    device = torch.device(cfg['device'])
    yolo_model = model.create_model(cfg)



    # Train model
    if args.mode == 'train':
        logger.info("Starting training...")
        yolo_model = model.model_to_device(yolo_model, device)
        train.train_model(yolo_model, None, None, cfg, device)  # YOLOv8 doesn't need separate loaders
    
    # Evaluate model
    elif args.mode == 'evaluate':
        logger.info("Starting evaluation...")
        yolo_model = model.best_model(cfg)
        yolo_model = model.model_to_device(yolo_model, device)
        evaluate.evaluate_model(yolo_model, None, cfg, device)  # YOLOv8 doesn't need separate loaders
    
    # Run inference
    elif args.mode == 'inference':
        logger.info("Running inference...")
        yolo_model = model.best_model(cfg)
        yolo_model = model.model_to_device(yolo_model, device)
        results = evaluate.run_inference(yolo_model, args.image_path, cfg, device)
        utils.visualize_results( args.image_path, results, cfg['class_names'])
    
    else:
        logger.error(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection using YOLOv8")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'inference'], required=True, help='Mode to run the script in')
    parser.add_argument('--image_path', type=str, help='Path to image for inference')
    args = parser.parse_args()
    
    main(args)