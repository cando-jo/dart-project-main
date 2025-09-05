import os
import pydicom
from PIL import Image
import numpy as np

image_directory_path = r"C:\Users\RPitchuka4915\Downloads\OAIData\extracted\00m\0.C.2"

for filename in os.listdir(image_directory_path):
    filepath = os.path.join(image_directory_path, filename)

    if filename.lower().endswith('.dcm'):
        try:
            # Read DICOM file
            dicom_file_data = pydicom.dcmread(filepath)
            image_array = dicom_file_data.pixel_array
            
            # Normalize to 0–255
            image = image_array.astype(np.float32)
            image -= image.min()
            if image.max() > 0:
                image /= image.max()
            image *= 255.0
            image = image.astype(np.uint8)

            # Convert to PIL Image
            image_pil = Image.fromarray(image)

            # Replace .dcm with .png in the same folder
            new_filename = filename.replace('.dcm', '.png')
            output_path = os.path.join(image_directory_path, new_filename)

            # Save PNG
            image_pil.save(output_path)

            # Optionally delete original .dcm file
            os.remove(filepath)

            print(f"Converted {filename} → {new_filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
