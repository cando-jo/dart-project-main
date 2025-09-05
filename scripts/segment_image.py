import os
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from sklearn.metrics import jaccard_score

# Initialize SAM model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

# Load YOLO model
yolo_model = YOLO("runs/detect/train5/weights/best.pt")

# Path to dataset
image_folder_path = "data/Promise12MSBench/test/images"

# Function to calculate Intersection over Union (IoU)
def calculate_iou(pred_mask, gt_mask):
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask) - intersection
    iou_score = intersection / union
    return iou_score

# Function to calculate Dice Score
def calculate_dice(pred_mask, gt_mask):
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    dice_score = (2 * intersection) / (union + intersection)
    return dice_score

# Variables to store the total IoU and Dice scores
iou_total = []
dice_total = []

# Loop through all images in the dataset
for file in os.listdir(image_folder_path):
        
        path_to_image = os.path.join(image_folder_path, file)
        image = cv2.imread(path_to_image)

        # Convert to RGB for SAM Input
        sam_input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(sam_input_image)

        # Run YOLO to get bounding boxes for the image
        yolo_results = yolo_model.predict(path_to_image)
        bboxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
        scores = yolo_results[0].boxes.conf.cpu().numpy()

        # Select the best bounding box (highest confidence)
        best_bbox = bboxes[np.argmax(scores)]
    
        # Get the corresponding ground truth mask
        gt_mask_path = path_to_image.replace('images', 'masks').replace('image', 'mask')
        gt_mask = cv2.imread(gt_mask_path)
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)

        # Get the best predicted masks from SAM for the best bbox
        sam_masks, sam_scores, _ = predictor.predict(box = best_bbox, multimask_output=True)
        pred_mask = sam_masks[np.argmax(sam_scores)]  
        pred_mask = pred_mask.astype(np.uint8)*255

        # Save the mask overlayed image 
        # overlay_pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
        # overlayed_image = cv2.addWeighted(image, 1, overlay_pred_mask, 0.4, 0)
        # cv2.imshow("Predicted Mask", overlayed_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Calculate IoU and Dice score for this image
        # iou_score = calculate_iou(pred_mask, gt_mask)
        dice_score = calculate_dice(pred_mask, gt_mask)

        # Append the scores to the totals
        # iou_total.append(iou_score)
        dice_total.append(dice_score)

        print(f"Processed {file}: Dice = {dice_score:.4f}")

# Calculate overall IoU and Dice score
# average_iou = np.mean(iou_total)
average_dice = np.mean(dice_total)

# print(f"\nOverall IoU: {average_iou:.4f}")
print(f"Overall Dice Score: {average_dice:.4f}")
