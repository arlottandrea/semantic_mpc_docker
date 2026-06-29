import numpy as np
import cv2
from typing import List, Union


def get_random_color(seed: int = None) -> tuple[int, int, int]:
    """
    Returns a fixed, bright light-green color.

    The 'seed' argument is ignored but kept for compatibility
    if other parts of the code expect it.

    Returns:
        tuple[int, int, int]: The BGR tuple for a bright green color.
    """
    # BGR color for a bright, light green
    return (100, 255, 100)


def draw_detections(img: np.array, bboxes: List[List[int]], classes: List[int],
                    class_labels: Union[List[str], None], confs: List[int]):
    """
    Draws bounding boxes and labels on an image using a bright green color.

    Args:
        img (np.array): The image to draw on.
        bboxes (List[List[int]]): A list of bounding boxes, each as [x1, y1, x2, y2].
        classes (List[int]): A list of class IDs for each bounding box.
        class_labels (Union[List[str], None]): A list of string labels for classes.
        confs (List[int]): A list of confidence scores for each detection.

    Returns:
        np.array: The image with detections drawn on it.
    """
    # Get the single color to use for all detections
    color = get_random_color()

    for bbox, cls, conf in zip(bboxes, classes, confs):
        x1, y1, x2, y2 = bbox

        # Draw the bounding box
        img = cv2.rectangle(
            img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3
        )

        # Draw the label with bigger, bolder text
        if class_labels:
            label = (f'{class_labels[int(cls)]} {conf:.2f}')
            x_text = int(x1)
            # Adjust y-position to be above the box
            y_text = max(15, int(y1 - 10))

            img = cv2.putText(
                img, label, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.9,  # Increased font scale for bigger text
                color=color,
                thickness=2,    # Increased thickness for bolder text
                lineType=cv2.LINE_AA
            )

    return img