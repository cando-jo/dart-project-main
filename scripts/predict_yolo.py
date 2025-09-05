import os
from ultralytics import YOLO

def predict_bbox():
   
   image_path = 'data/Promise12MSBench/test/images/'
   model = YOLO('runs/detect/train5/weights/best.pt')
   for file in os.listdir(image_path):
      image_file = os.path.join(image_path, file)
      results = model.predict(image_file)
   return model, image_file, results



