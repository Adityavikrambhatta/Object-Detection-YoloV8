import logging
import torch
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np 
def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# Utility function to draw rounded rectangles
def draw_rounded_rectangle(image, pt1, pt2, color, thickness, radius=0.2):
    x1, y1 = pt1
    x2, y2 = pt2

    width = x2 - x1
    height = y2 - y1
    radius = int(min(width, height) * radius)

    # Draw the rectangle with rounded corners
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

    # Draw four arcs (for rounded corners)
    cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

# Utility function for drawing a transparent bounding box
def draw_transparent_box(image, pt1, pt2, color, alpha=0.4):
    overlay = image.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    # Apply transparency to the image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def visualize_results(image_path, prediction, class_names):

    image = cv2.imread(image_path)

    color = {0: (0,0,139), 1 : (173, 216, 230)}
    # Ensure the image is contiguous and writable
    image = np.ascontiguousarray(image)

    for result in prediction[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        # Convert to integers for OpenCV
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw a transparent bounding box
        draw_transparent_box(image, (x1, y1), (x2, y2), color[int(class_id)], alpha=0.3)

        # Draw a rounded rectangle around the box
        draw_rounded_rectangle(image, (x1, y1), (x2, y2), color[int(class_id)], thickness=2, radius=0.15)

        # Create a label with class name and score
        label = f'{prediction[0].names[int(class_id)]} {score:.2f}'

        # Get the text size
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

        # Draw a filled rectangle behind the text for better readability
        cv2.rectangle(image, (x1, y1 - h - 5), (x1 + w, y1), color[int(class_id)], -1)

        # Add the label text
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the final image
    cv2.imshow("YOLOv8 Aesthetic Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the output image
    cv2.imwrite('predictions/', image)