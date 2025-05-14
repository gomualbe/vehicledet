import cv2
import os
import numpy as np
from argparse import ArgumentParser

main_dir = os.path.dirname(os.path.abspath(__file__))

parser = ArgumentParser(description="Resize images and labels")
parser.add_argument("--img_dir", type=str, help="Directory of images to resize")
parser.add_argument("--label_dir", type=str, help="Directory of labels to resize")
parser.add_argument("--target_size", type=int, default=640, help="Target size for resizing images")

def resize_image(image, target_size):
  ''' Resize an image to a target size '''
  if image is not None:
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def adjust_labels(label_path, original_size, target_size):
    """Adjust labels to match new image size and preserve YOLO format."""
    with open(label_path, "r") as f:
        lines = f.readlines()

    orig_w, orig_h = original_size
    target_w, target_h = target_size

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    new_lines = []
    for line in lines:
        parts = list(map(float, line.strip().split()))
        cls, x, y, w, h = parts

        # Convert from normalized to absolute coordinates
        abs_x = x * orig_w
        abs_y = y * orig_h
        abs_w = w * orig_w
        abs_h = h * orig_h

        # Resize in absolute coordinates
        abs_x *= scale_x
        abs_y *= scale_y
        abs_w *= scale_x
        abs_h *= scale_y

        # Convert back to normalized coordinates
        x = abs_x / target_w
        y = abs_y / target_h
        w = abs_w / target_w
        h = abs_h / target_h

        new_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    with open(label_path, "w") as f:
        f.writelines(new_lines)
        
if __name__ == "__main__":
    args = parser.parse_args()

    img_dir = args.img_dir
    label_dir = args.label_dir
    target_size = (args.target_size, args.target_size)

    if not os.path.exists(img_dir):
        print(f"Image directory {img_dir} does not exist.")
        exit(1)

    if not os.path.exists(label_dir):
        print(f"Label directory {label_dir} does not exist.")
        exit(1)

    # Resize images and adjust labels
    for img_name in os.listdir(img_dir):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(img_dir, img_name)
            label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))

            # Read and resize image
            image = cv2.imread(img_path)
            original_size = (image.shape[1], image.shape[0])  # (width, height)
            resized_image = resize_image(image, target_size)

            # Save resized image
            cv2.imwrite(img_path, resized_image)

            # Adjust labels if they exist
            if os.path.exists(label_path):
                adjust_labels(label_path, original_size, target_size)