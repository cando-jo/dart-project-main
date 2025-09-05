import os
import cv2

def mask_to_yolo_bbox(mask, img_shape):
    h, w = img_shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yolo_bboxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        yolo_bboxes.append([0, x_center, y_center, bw / w, bh / h])
    return yolo_bboxes

def generate_yolo_annotations(base_dir):
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(base_dir, split, "images")
        msk_dir = os.path.join(base_dir, split, "masks")
        lbl_dir = os.path.join(base_dir, split, "labels")
        os.makedirs(lbl_dir, exist_ok=True)

        for mask in sorted(os.listdir(msk_dir)):
            if not mask.endswith(".png"):
                continue

            image_name = mask.replace("mask", "image")
            mask_path = os.path.join(msk_dir, mask)
            image_path = os.path.join(img_dir, image_name)
            label_path = os.path.join(lbl_dir, image_name.replace(".png", ".txt"))

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(image_path)
            if mask is None or image is None:
                continue
            h, w = image.shape[:2]
            bboxes = mask_to_yolo_bbox(mask, (h, w))
            with open(label_path, "w") as f:
                for bbox in bboxes:
                    f.write(" ".join(map(str, bbox)) + "\n")

generate_yolo_annotations("data/Promise12MSBench")
